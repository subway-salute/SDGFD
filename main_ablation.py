import argparse, os, random, wandb, torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from construct_loader import construct_loader
from module import LegoSOTANet, CNN_Frontend
from losses import SupConLoss


def set_seed(seed=42):
    random.seed(seed);
    os.environ['PYTHONHASHSEED'] = str(seed);
    np.random.seed(seed)
    torch.manual_seed(seed);
    torch.cuda.manual_seed(seed);
    torch.backends.cudnn.deterministic = True


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=3300.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--variant', type=str, required=True, choices=['A', 'B', 'C', 'D'],
                        help='A: Pure CNN, B: +Wavelet, C: +Wavelet+ADAIN, D: Full(OURS)')
    args = parser.parse_args()
    set_seed(42);
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(project="PU_Ablation_Study", name=f"Var{args.variant}_{args.source}", config=vars(args))
    train_loader = construct_loader('./data', 'PU', args.source, 40, True)
    test_loaders = [construct_loader('./data', 'PU', t, 40, False) for t in args.target.split(',')]

    # ================= 动态消融架构解析 =================
    use_wavelet = args.variant in ['B', 'C', 'D']
    use_adain = args.variant in ['C', 'D']
    use_supcon = args.variant == 'D'

    frontend_type = 'wavelet' if use_wavelet else 'cnn'
    net = LegoSOTANet(frontend=frontend_type).to(device)

    criterion_cls = nn.CrossEntropyLoss();
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_sup = SupConLoss(temperature=0.07).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    iter_num = 0;
    tail_acc, tail_f1, tail_auc = [], [], []

    for epoch in range(1, 41):
        net.train();
        total_loss = 0.0
        for inputs, labels in train_loader:
            B = inputs.size(0);
            inputs, labels = inputs.to(device).float(), labels.to(device)
            if len(inputs.shape) == 2: inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()

            iter_num += 1
            alpha = calc_coeff(iter_num, max_iter=3300.0)

            f_ori = net.extract_FE(inputs)

            if use_adain:
                z = (0.05 + 1.90 * torch.rand(B, 1)).to(device);
                y = torch.randn(B, 1).to(device)
                f_lat_cls = net.latent_gen(f_ori, z, y)
                di_cls = net.DI(torch.cat([f_ori, f_lat_cls], 0))
                logits_cls = net.classifier(di_cls)

                l_ori, l_lat = torch.split(logits_cls, B)
                loss = criterion_cls(l_ori, labels) + criterion_cls(l_lat, labels)

                if use_supcon:
                    proj_cls = net.projector(torch.cat([f_ori, f_lat_cls], 0))
                    p_ori, p_lat = torch.split(proj_cls, B)
                    loss += 0.1 * criterion_sup(torch.stack([p_ori, p_lat], 1), labels)

                # 对抗域损失
                f_lat_adv = net.latent_gen(f_ori.detach(), z, y)
                di_adv = net.DI(torch.cat([f_ori, f_lat_adv], 0))
                d_ori, d_lat = torch.split(di_adv, B)
                o_ori = net.batchnorm_D(
                    torch.bmm(F.softmax(l_ori.detach(), 1).unsqueeze(2), d_ori.unsqueeze(1)).view(B, -1))
                o_lat = net.batchnorm_D(
                    torch.bmm(F.softmax(l_lat.detach(), 1).unsqueeze(2), d_lat.unsqueeze(1)).view(B, -1))
                loss += 1.0 * (criterion_bce(net.discriminator(o_ori, alpha), torch.zeros(B, 1).to(device)) +
                               criterion_bce(net.discriminator(o_lat, alpha), torch.ones(B, 1).to(device)))
            else:
                # 不使用对抗和风格迁移时，只走基础分类通路
                di_cls = net.DI(f_ori)
                logits_cls = net.classifier(di_cls)
                loss = criterion_cls(logits_cls, labels)

            loss.backward();
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            optimizer.step();
            total_loss += loss.item()

        scheduler.step()

        # ================= 测试 =================
        net.eval();
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for t_loader in test_loaders:
                for inputs, labels in t_loader:
                    inputs = inputs.to(device).float()
                    if len(inputs.shape) == 2: inputs = inputs.unsqueeze(1)

                    logits = net.classifier(net.DI(net.extract_FE(inputs)))

                    all_labels.extend(labels.numpy())
                    all_preds.extend(logits.argmax(1).cpu().numpy())
                    all_probs.extend(F.softmax(logits, dim=1).cpu().numpy())

        acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = 100. * f1_score(all_labels, all_preds, average='macro')
        auc = 100. * roc_auc_score(all_labels, all_probs, multi_class='ovr')

        if epoch > 30: tail_acc.append(acc); tail_f1.append(f1); tail_auc.append(auc)
        wandb.log({"Epoch": epoch, "Loss": total_loss / len(train_loader), "Acc": acc, "F1": f1, "AUC": auc})

    print(f"FINAL_ACCURACY:{np.mean(tail_acc):.2f}")
    wandb.log(
        {"Final_Avg_Acc": np.mean(tail_acc), "Final_Avg_F1": np.mean(tail_f1), "Final_Avg_AUC": np.mean(tail_auc)})
    wandb.finish()


if __name__ == '__main__':
    main()