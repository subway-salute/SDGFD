import argparse, os, random, wandb, torch, pywt
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from construct_loader import construct_loader


def set_seed(seed=42):
    random.seed(seed);
    os.environ['PYTHONHASHSEED'] = str(seed);
    np.random.seed(seed)
    torch.manual_seed(seed);
    torch.cuda.manual_seed(seed);
    torch.backends.cudnn.deterministic = True


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
        mu = x.mean(dim=2, keepdim=True);
        var = x.var(dim=2, keepdim=True)
        x_normed = (x - mu.detach()) / (var + 0.1).sqrt().detach()
        gamma = self.fc1(s1).view(s1.size(0), -1, 1);
        beta = self.fc2(s2).view(s2.size(0), -1, 1)
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
        self.w1 = DWConv(num_channels=1);
        self.c1 = SConv_1D(2, 16, 3)
        self.w2 = DWConv(num_channels=16);
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
        self.G_te = Fea_Extraction(is_teacher=True);
        self.C_te = nn.Linear(32, num_classes)
        self.G_st = Fea_Extraction(is_teacher=False);
        self.C_st = nn.Linear(32, num_classes)


def compute_kl_loss(p, q, T=3):
    return (F.kl_div(F.log_softmax(p / T, dim=-1), F.softmax(q / T, dim=-1), reduction='batchmean') +
            F.kl_div(F.log_softmax(q / T, dim=-1), F.softmax(p / T, dim=-1), reduction='batchmean')) / 2


def F_distance(x, y): return torch.norm(x - y).mean()


def gram(y): return y.bmm(y.transpose(1, 2)) / (y.size(1) * y.size(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()
    set_seed(42);
    device = torch.device('cuda')

    wandb.init(project="PU_Final_6Models", name=f"DWCN_{args.source}", config=vars(args))
    train_loader = construct_loader('./data', 'PU', args.source, 40, True)

    target_list = args.target.split(',')
    test_loaders = [construct_loader('./data', 'PU', t, 40, False) for t in target_list]

    model = DWCN().to(device)
    criterion = nn.CrossEntropyLoss()
    opt_te = optim.Adam(list(model.G_te.parameters()) + list(model.C_te.parameters()), lr=0.01)
    opt_st = optim.Adam(list(model.G_st.parameters()) + list(model.C_st.parameters()), lr=0.01)
    opt_ld = optim.Adam(model.G_te.dp.parameters(), lr=0.01)

    tail_acc, tail_f1, tail_auc = [], [], []

    for epoch in range(1, 41):
        model.train();
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device)
            if len(inputs.shape) == 2: inputs = inputs.unsqueeze(1)
            opt_te.zero_grad();
            opt_st.zero_grad()

            f_te = model.G_te(inputs, perturb=True);
            f_st = model.G_st(inputs)
            s_te, s_st = model.C_te(f_te), model.C_st(f_st)
            loss_c = (criterion(s_te, labels) + criterion(s_st, labels)) * 0.5
            loss_kl = compute_kl_loss(s_st, s_te)

            (loss_c + loss_kl).backward();
            opt_te.step();
            opt_st.step()

            _ = model.G_te(inputs, perturb=True);
            _ = model.G_st(inputs)
            loss_ccp = 0.05 * (-F_distance(gram(model.G_st.l0), gram(model.G_te.l0)))
            opt_ld.zero_grad();
            loss_ccp.backward();
            opt_ld.step()
            total_loss += loss_c.item()

        # ================= 混合目标域测试 & 特征提取 =================
        model.eval();
        all_labels, all_preds, all_probs, all_features = [], [], [], []
        with torch.no_grad():
            for t_loader in test_loaders:
                for inputs, labels in t_loader:
                    inputs = inputs.to(device).float()
                    if len(inputs.shape) == 2: inputs = inputs.unsqueeze(1)

                    # 剥离特征和分类结果
                    features = model.G_st(inputs)
                    logits = model.C_st(features)

                    all_features.extend(features.cpu().numpy())
                    all_labels.extend(labels.numpy())
                    all_preds.extend(logits.argmax(1).cpu().numpy())
                    all_probs.extend(F.softmax(logits, dim=1).cpu().numpy())

        acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = 100. * f1_score(all_labels, all_preds, average='macro')
        auc = 100. * roc_auc_score(all_labels, all_probs, multi_class='ovr')

        if epoch > 30:
            tail_acc.append(acc);
            tail_f1.append(f1);
            tail_auc.append(auc)

        wandb.log({"Epoch": epoch, "Loss": total_loss / len(train_loader), "Acc": acc, "F1": f1, "AUC": auc})

        # 🌟 最后一轮保存画图数据 🌟
        if epoch == 40:
            os.makedirs("plot_data", exist_ok=True)
            np.save(f"plot_data/features_DWCN_{args.source}.npy", np.array(all_features))
            np.save(f"plot_data/labels_DWCN_{args.source}.npy", np.array(all_labels))
            np.save(f"plot_data/preds_DWCN_{args.source}.npy", np.array(all_preds))

    print(f"FINAL_ACCURACY:{np.mean(tail_acc):.2f}")
    wandb.log(
        {"Final_Avg_Acc": np.mean(tail_acc), "Final_Avg_F1": np.mean(tail_f1), "Final_Avg_AUC": np.mean(tail_auc)})
    wandb.finish()


if __name__ == '__main__':
    main()