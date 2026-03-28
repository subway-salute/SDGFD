import argparse, os, random, wandb, torch
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


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=3300.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad): return -coeff * grad.clone()

    return fun1


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features.squeeze(), dim=1).unsqueeze(1)
        device = features.device
        features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = 1

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1.0, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        loss = - (self.temperature / 0.07) * mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()


class ADAIN(nn.Module):
    def __init__(self):
        super(ADAIN, self).__init__()
        self.norm = nn.InstanceNorm1d(1, affine=False)
        self.fc_layer1 = nn.Linear(1, 256);
        self.fc_layer2 = nn.Linear(1, 256)

    def forward(self, x, z, y):
        x = x.unsqueeze(1)
        gamma = self.fc_layer1(z).unsqueeze(1);
        beta = self.fc_layer2(y).unsqueeze(1)
        return (gamma * self.norm(x) + beta).squeeze(1)


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size=300):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU();
        self.dropout1 = nn.Dropout(0.5);
        self.dropout2 = nn.Dropout(0.5)
        self.iter_num = 0;
        self.max_iter = 3300.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
            coeff = calc_coeff(self.iter_num, 1.0, 0.0, 10, self.max_iter)
            x = x * 1.0;
            x.register_hook(grl_hook(coeff))
        x = self.dropout2(self.dropout1(self.relu1(self.ad_layer1(x))))
        return self.ad_layer3(x)


class GFCD(nn.Module):
    def __init__(self, num_classes=3, end_feat=256):
        super(GFCD, self).__init__()
        self.FE = nn.Sequential(
            nn.Conv1d(1, 16, 7, 2, 3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(3, 2, 1),
            nn.Conv1d(16, 64, 3, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, end_feat, 3, 2, 1), nn.AdaptiveAvgPool1d(1)
        )
        self.DI = nn.Sequential(nn.Linear(end_feat, end_feat), nn.ReLU(), nn.Dropout(0.5))
        self.CL = nn.Linear(end_feat, num_classes)
        self.adain = ADAIN()
        self.batchnorm = nn.BatchNorm1d(end_feat * num_classes)
        self.D = AdversarialNetwork(end_feat * num_classes)

    def forward(self, x, z=None, y=None):
        feat = self.FE(x).squeeze(2)
        if self.training:
            feat_new1 = self.adain(feat, z, y)
            feat_new2 = self.adain(feat.detach().clone(), z, y)
            x_class = torch.cat([feat, feat_new1], 0);
            x_domain = torch.cat([feat, feat_new2], 0)
            x_di_class = self.DI(x_class);
            x_di_domain = self.DI(x_domain)
            cls_out = self.CL(x_di_class)
            op_out = torch.bmm(cls_out.detach().unsqueeze(2), x_di_domain.unsqueeze(1))
            pred_domain = self.D(self.batchnorm(op_out.view(op_out.size(0), -1)))
            return x_di_class, cls_out, pred_domain
        else:
            x_di = self.DI(feat)
            cls_out = self.CL(x_di)
            op_out = torch.bmm(cls_out.unsqueeze(2), x_di.unsqueeze(1))
            pred_domain = self.D(op_out.view(op_out.size(0), -1))
            # 必须返回三个元素
            return x_di, cls_out, pred_domain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()
    set_seed(42);
    device = torch.device('cuda')

    wandb.init(project="PU_Thesis_Final", name=f"DACN_{args.source}", config=vars(args))
    train_loader = construct_loader('./data', 'PU', args.source, 40, True)

    target_list = args.target.split(',')
    test_loaders = [construct_loader('./data', 'PU', t, 40, False) for t in target_list]

    net = GFCD().to(device)
    criterion_cls = nn.CrossEntropyLoss();
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_sup = SupConLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    tail_acc, tail_f1, tail_auc = [], [], []

    for epoch in range(1, 41):
        net.train();
        total_loss = 0.0
        for inputs, labels in train_loader:
            B = inputs.size(0);
            inputs, labels = inputs.to(device).float(), labels.to(device)
            if len(inputs.shape) == 2: inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()

            z = (0.05 + 1.90 * torch.rand(B, 1)).to(device);
            y = torch.randn(B, 1).to(device)
            x_di, cls_out, pred_domain = net(inputs, z, y)
            cls_ori, cls_new = torch.split(cls_out, B);
            di_ori, di_new = torch.split(x_di, B)

            loss_c = criterion_cls(cls_ori, labels) + criterion_cls(cls_new, labels)
            loss_sup = criterion_sup(torch.stack([di_ori, di_new], dim=1), labels)
            dom_labels = torch.cat([torch.zeros(B, 1), torch.ones(B, 1)], 0).to(device)
            loss_adv = criterion_bce(pred_domain, dom_labels)

            loss = loss_c + 0.1 * loss_sup + 1.0 * loss_adv
            loss.backward();
            optimizer.step();
            total_loss += loss.item()

        # ================= 混合目标域测试 & 特征提取 =================
        net.eval();
        all_labels, all_preds, all_probs, all_features = [], [], [], []
        with torch.no_grad():
            for t_loader in test_loaders:
                for inputs, labels in t_loader:
                    inputs = inputs.to(device).float()
                    if len(inputs.shape) == 2: inputs = inputs.unsqueeze(1)

                    # 修复 DACN 测试模式下返回元组的问题
                    features, logits, _ = net(inputs)

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
            np.save(f"plot_data/features_DACN_{args.source}.npy", np.array(all_features))
            np.save(f"plot_data/labels_DACN_{args.source}.npy", np.array(all_labels))
            np.save(f"plot_data/preds_DACN_{args.source}.npy", np.array(all_preds))

    print(f"FINAL_ACCURACY:{np.mean(tail_acc):.2f}")
    wandb.log(
        {"Final_Avg_Acc": np.mean(tail_acc), "Final_Avg_F1": np.mean(tail_f1), "Final_Avg_AUC": np.mean(tail_auc)})
    wandb.finish()


if __name__ == '__main__':
    main()