import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import random

from construct_loader import construct_loader
from module import LegoSOTANet
from losses import SupConLoss


def set_deterministic_seed(seed=42):
    """最严苛的随机种子锁定，保证100%可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='PU')
    parser.add_argument('--source', type=str, default='N15_M01')  # 最难源域
    parser.add_argument('--target', type=str, default='N09_M07')  # 最难目标域
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--epoch', type=int, default=100)

    # 动态架构参数
    parser.add_argument('--frontend', type=str, choices=['cnn', 'wavelet'], default='wavelet')
    parser.add_argument('--mutation', type=str, choices=['adain', 'res_adain'], default='res_adain')
    parser.add_argument('--align', type=str, choices=['none', 'supcon'], default='supcon')
    parser.add_argument('--adv', type=str, choices=['none', 'dann', 'cdan'], default='cdan')

    args = parser.parse_args()
    set_deterministic_seed(42)  # 锁定随机性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = construct_loader(args.data_root, args.dataset, args.source, args.batch_size, True)
    test_loader = construct_loader(args.data_root, args.dataset, args.target, args.batch_size, False)

    net = LegoSOTANet(frontend=args.frontend, mutation=args.mutation, adv=args.adv).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_bce = nn.BCEWithLogitsLoss()
    # 🌟 补丁 1：SupCon 降温防爆 (temperature 调至 0.5)
    criterion_sup = SupConLoss(temperature=0.5).to(device)

    optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    total_iters = args.epoch * len(train_loader)
    iter_num = 0
    tail_acc_list = []

    for epoch in range(1, args.epoch + 1):
        net.train()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            B = inputs.size(0)
            inputs, labels = inputs.to(device), labels.to(device)

            # 🌟 补丁 2：输入张量自适应塑形 (防止预处理丢失通道维度)
            inputs = inputs.float()
            if len(inputs.shape) == 2:
                inputs = inputs.unsqueeze(1)

            optimizer.zero_grad()

            p = float(iter_num) / total_iters
            # 🌟 补丁 3：平滑对抗烈度，给主网络喘息空间 (-10 改为 -5)
            alpha = 2.0 / (1.0 + np.exp(-5 * p)) - 1.0
            iter_num += 1

            # 1. 提特征 & 变异
            feat_ori = net.extract_FE(inputs)
            z, y = torch.rand(B, 1).to(device), torch.randn(B, 1).to(device)
            feat_latent = net.latent_gen(feat_ori, z, y)

            # 2. 联合推理
            feat_all = torch.cat([feat_ori, feat_latent], dim=0)
            di_all = net.DI(feat_all)
            logits_all = net.classifier(di_all)
            proj_all = net.projector(feat_all)

            logits_ori, logits_latent = torch.split(logits_all, B)
            di_ori, di_latent = torch.split(di_all, B)
            proj_ori, proj_latent = torch.split(proj_all, B)

            # 3. 计算 基础分类 Loss
            loss = criterion_cls(logits_ori, labels) + criterion_cls(logits_latent, labels)

            # 4. 动态组装 Align Loss (SupCon)
            if args.align == 'supcon':
                loss_sup = criterion_sup(torch.stack([proj_ori, proj_latent], dim=1), labels)
                # 🌟 补丁 4：降低 SupCon 权重，防止引力过大导致崩溃
                loss += 0.05 * loss_sup

                # 5. 动态组装 Adv Loss (DANN / CDAN)
            if args.adv != 'none':
                domain_label_ori = torch.zeros(B, 1).to(device)
                domain_label_latent = torch.ones(B, 1).to(device)

                if args.adv == 'cdan':
                    prob_ori = F.softmax(logits_ori.detach(), dim=1)
                    prob_latent = F.softmax(logits_latent.detach(), dim=1)
                    op_ori = net.batchnorm_D(torch.bmm(prob_ori.unsqueeze(2), di_ori.unsqueeze(1)).view(B, -1))
                    op_latent = net.batchnorm_D(torch.bmm(prob_latent.unsqueeze(2), di_latent.unsqueeze(1)).view(B, -1))
                else:  # dann
                    op_ori, op_latent = di_ori, di_latent

                pred_d_ori = net.discriminator(op_ori, alpha=alpha)
                pred_d_latent = net.discriminator(op_latent, alpha=alpha)
                loss_adv = criterion_bce(pred_d_ori, domain_label_ori) + criterion_bce(pred_d_latent,
                                                                                       domain_label_latent)
                loss += 1.0 * loss_adv

            loss.backward()

            # 🌟 补丁 5：物理防爆锁，强制截断爆炸梯度
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)

            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # =================== 测试阶段 ===================
        net.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device).float()

                # 测试集同样需要塑形保护
                if len(inputs.shape) == 2:
                    inputs = inputs.unsqueeze(1)

                logits = net.classifier(net.DI(net.extract_FE(inputs)))
                all_labels.extend(labels.numpy())
                all_preds.extend(logits.argmax(1).cpu().numpy())

        test_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))

        # 记录最后 10 轮用于求稳健平均值
        if epoch > args.epoch - 10:
            tail_acc_list.append(test_acc)

        print(f"Epoch [{epoch:03d}/{args.epoch}] | Loss: {total_loss / len(train_loader):.4f} | Acc: {test_acc:.2f}%")

    avg_tail_acc = np.mean(tail_acc_list)
    print(f"FINAL_ACCURACY:{avg_tail_acc:.2f}")


if __name__ == '__main__':
    main()