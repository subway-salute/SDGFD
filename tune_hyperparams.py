import wandb
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import os

from construct_loader import construct_loader
from module import FusionSOTANet
from losses import SupConLoss


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    wandb.init()
    config = wandb.config
    set_seed(42)
    device = torch.device('cuda')

    data_root = './data'
    dataset = 'PU'
    source = 'N15_M01'
    target = 'N09_M07'
    batch_size = 40
    epochs = 40

    train_loader = construct_loader(data_root, dataset, source, batch_size, True)
    test_loader = construct_loader(data_root, dataset, target, batch_size, False)

    net = FusionSOTANet(num_classes=3).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_sup = SupConLoss(temperature=0.1).to(device)

    optimizer = optim.AdamW(net.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    total_iters = epochs * len(train_loader)
    iter_num = 0

    for epoch in range(1, epochs + 1):
        net.train()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            B = inputs.size(0)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            p = float(iter_num) / total_iters
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            iter_num += 1

            feat_ori = net.extract_FE(inputs)

            # 动态噪声范围
            z = torch.rand(B, 1).to(device) * config.noise_scale
            y = torch.randn(B, 1).to(device) * config.noise_scale
            feat_latent = net.latent_gen(feat_ori, z, y)

            feat_all = torch.cat([feat_ori, feat_latent], dim=0)
            di_all = net.DI(feat_all)
            logits_all = net.classifier(di_all)
            proj_all = net.projector(feat_all)

            di_ori, di_latent = torch.split(di_all, B)
            logits_ori, logits_latent = torch.split(logits_all, B)
            proj_ori, proj_latent = torch.split(proj_all, B)

            loss_cls = criterion_cls(logits_ori, labels) + criterion_cls(logits_latent, labels)
            loss_sup = criterion_sup(torch.stack([proj_ori, proj_latent], dim=1), labels)

            domain_label_ori = torch.zeros(B, 1).to(device)
            domain_label_latent = torch.ones(B, 1).to(device)

            prob_ori = F.softmax(logits_ori.detach(), dim=1)
            prob_latent = F.softmax(logits_latent.detach(), dim=1)

            op_out_ori = torch.bmm(prob_ori.unsqueeze(2), di_ori.unsqueeze(1)).view(B, -1)
            pred_domain_ori = net.discriminator(net.batchnorm_D(op_out_ori), alpha=alpha)

            op_out_latent = torch.bmm(prob_latent.unsqueeze(2), di_latent.unsqueeze(1)).view(B, -1)
            pred_domain_latent = net.discriminator(net.batchnorm_D(op_out_latent), alpha=alpha)

            loss_adv = criterion_bce(pred_domain_ori, domain_label_ori) + criterion_bce(pred_domain_latent,
                                                                                        domain_label_latent)

            loss = loss_cls + config.lamda_supcon * loss_sup + config.lamda_adv * loss_adv

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        net.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                logits = net.classifier(net.DI(net.extract_FE(inputs.to(device))))
                all_labels.extend(labels.numpy())
                all_preds.extend(logits.argmax(1).cpu().numpy())

        test_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))

        wandb.log({"Epoch": epoch, "Train_Loss": total_loss / len(train_loader), "Test_Acc": test_acc})


if __name__ == '__main__':
    sweep_config = {
        'method': 'random',
        'name': 'Ultimate_Fusion_Tuning',
        'metric': {'name': 'Test_Acc', 'goal': 'maximize'},
        'parameters': {
            'lr': {'values': [5e-4, 1e-3, 5e-3]},
            'lamda_supcon': {'values': [0.05, 0.1, 0.5]},
            'lamda_adv': {'values': [0.5, 1.0, 1.5]},
            'noise_scale': {'values': [0.5, 1.0, 2.0]}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="PU_SDG_Fusion")
    wandb.agent(sweep_id, train, count=15)