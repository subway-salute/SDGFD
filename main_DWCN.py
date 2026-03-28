import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import wandb
import os
import random
import pywt

from construct_loader import construct_loader


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class SConv_1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU())

    def forward(self, x): return self.conv(x)


class LP(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.fc1 = nn.Linear(style_dim, num_features)
        self.fc2 = nn.Linear(style_dim, num_features)

    def forward(self, x, s1, s2):
        mu = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True)
        x_normed = (x - mu.detach()) / (var + 0.1).sqrt().detach()
        gamma = self.fc1(s1).view(s1.size(0), -1, 1)
        beta = self.fc2(s2).view(s2.size(0), -1, 1)
        # DWCN 原版带残差的扰动
        return x + (gamma * x_normed + beta)


class DWConv(nn.Module):
    def __init__(self, wavelet='db4', num_channels=1):
        super().__init__()
        wavelet_obj = pywt.Wavelet(wavelet)
        l_filter, h_filter = wavelet_obj.filter_bank[0], wavelet_obj.filter_bank[1]
        self.kernel_size = len(l_filter)
        self.mWDN1 = nn.Parameter(torch.cat([
                                                torch.tensor(l_filter).float().unsqueeze(0).repeat(1, 1, 1),
                                                torch.tensor(h_filter).float().unsqueeze(0).repeat(1, 1, 1)
                                            ] * num_channels, dim=0))
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input):
        b, c, l = input.shape
        outsize = pywt.dwt_coeff_len(l, self.kernel_size, mode="zero")
        p = 2 * (outsize - 1) - l + self.kernel_size
        freq = F.conv1d(F.pad(input, (p // 2, p // 2 if p % 2 == 0 else p // 2 + 1)), self.mWDN1, groups=c, stride=2)
        return self.dropout(torch.cat([freq[:, ::2, :], freq[:, 1::2, :]], dim=1))


class Fea_Extraction(nn.Module):
    def __init__(self, is_teacher=False):
        super().__init__()
        self.is_teacher = is_teacher
        self.w1 = DWConv(num_channels=1)
        self.c1 = SConv_1D(2, 16, 3)
        self.w2 = DWConv(num_channels=16)
        self.c2 = SConv_1D(32, 32, 3)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dp = LP(16, 16) if is_teacher else None

    def forward(self, x, perturb=False):
        x1 = self.c1(self.w1(x))
        if perturb and self.is_teacher:
            x1 = self.dp(x1, torch.randn(len(x1), 16).to(x.device), torch.randn(len(x1), 16).to(x.device))
        self.l0 = x1
        x2 = self.c2(self.w2(x1))
        return self.avg_pool(x2).view(x2.size(0), -1)


class DWCN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.G_te = Fea_Extraction(is_teacher=True)
        self.C_te = nn.Linear(32, num_classes)
        self.G_st = Fea_Extraction(is_teacher=False)
        self.C_st = nn.Linear(32, num_classes)


def compute_kl_loss(p, q, T=3):
    return (F.kl_div(F.log_softmax(p / T, dim=-1), F.softmax(q / T, dim=-1), reduction='batchmean') +
            F.kl_div(F.log_softmax(q / T, dim=-1), F.softmax(p / T, dim=-1), reduction='batchmean')) / 2


def F_distance(x, y): return torch.norm(x - y).mean()


def gram(y): return y.bmm(y.transpose(1, 2)) / (y.size(1) * y.size(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='PU')
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--epoch', type=int, default=40)
    args = parser.parse_args()
    set_seed(42)
    device = torch.device('cuda')

    wandb.init(project="PU_SDG_Reproduce", name="DWCN_Original")
    train_loader = construct_loader(args.data_root, args.dataset, args.source, args.batch_size, True)
    test_loader = construct_loader(args.data_root, args.dataset, args.target, args.batch_size, False)

    model = DWCN().to(device)
    criterion = nn.CrossEntropyLoss()

    opt_te = optim.Adam(list(model.G_te.parameters()) + list(model.C_te.parameters()), lr=0.01)
    opt_st = optim.Adam(list(model.G_st.parameters()) + list(model.C_st.parameters()), lr=0.01)
    opt_ld = optim.Adam(model.G_te.dp.parameters(), lr=0.01)

    for epoch in range(1, args.epoch + 1):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            opt_te.zero_grad();
            opt_st.zero_grad()

            f_te = model.G_te(inputs, perturb=True)
            f_st = model.G_st(inputs)

            s_te, s_st = model.C_te(f_te), model.C_st(f_st)
            loss_c = (criterion(s_te, labels) + criterion(s_st, labels)) * 0.5
            loss_kl = compute_kl_loss(s_st, s_te)

            (loss_c + loss_kl).backward()
            opt_te.step();
            opt_st.step()

            _ = model.G_te(inputs, perturb=True)
            _ = model.G_st(inputs)
            loss_ccp = 0.05 * (-F_distance(gram(model.G_st.l0), gram(model.G_te.l0)))
            opt_ld.zero_grad()
            loss_ccp.backward()
            opt_ld.step()

            total_loss += loss_c.item()

        model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                preds = model.C_st(model.G_st(inputs.to(device)))
                all_labels.extend(labels.numpy())
                all_preds.extend(preds.argmax(1).cpu().numpy())

        test_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"DWCN | Epoch {epoch:02d} | Loss: {total_loss / len(train_loader):.4f} | Acc: {test_acc:.2f}%")
        wandb.log({"DWCN_Acc": test_acc})
    wandb.finish()


if __name__ == '__main__':
    main()

