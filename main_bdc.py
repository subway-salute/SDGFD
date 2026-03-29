import argparse, os, random, wandb, torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
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


# ==================== 100% 照搬原仓库 module.py 的核心网络与机制 ====================
class LSP_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(496, 1)
        self.fc2 = nn.Linear(496, 1)

    def forward(self, x):
        h1 = self.fc1(x)
        h2 = self.fc2(x)
        gamma = h1.view(h1.size(0), h1.size(1), 1)
        beta = h2.view(h2.size(0), h2.size(1), 1)
        return (1 + gamma) * x + beta


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1, 16, 32), nn.InstanceNorm1d(16), nn.ReLU(), nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(nn.Conv1d(16, 32, 3), nn.InstanceNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(nn.Conv1d(32, 64, 3), nn.InstanceNorm1d(64), nn.ReLU(), nn.MaxPool1d(4))
        self.conv4 = nn.Sequential(nn.Conv1d(64, 128, 3), nn.InstanceNorm1d(128), nn.ReLU(), nn.MaxPool1d(4))
        self.conv5 = nn.Sequential(nn.Conv1d(128, 128, 3), nn.InstanceNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))

    def forward(self, x, train=True):
        x1 = self.conv1(x)
        self.l0 = x1
        x2 = self.conv2(x1)
        self.l1 = x2
        x3 = self.conv3(x2)
        self.l2 = x3
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        fea = x5.view(x5.size(0), -1)
        return fea, x3


class CNN_Tea(nn.Module):
    def __init__(self):
        super(CNN_Tea, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1, 16, 32), nn.InstanceNorm1d(16), nn.ReLU(), nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(nn.Conv1d(16, 32, 3), nn.InstanceNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(nn.Conv1d(32, 64, 3), nn.InstanceNorm1d(64), nn.ReLU(), nn.MaxPool1d(4))
        self.conv4 = nn.Sequential(nn.Conv1d(64, 128, 3), nn.InstanceNorm1d(128), nn.ReLU(), nn.MaxPool1d(4))
        self.conv5 = nn.Sequential(nn.Conv1d(128, 128, 3), nn.InstanceNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.lsp_1 = LSP_1()

    def forward(self, x, perturb=False):
        x1 = self.conv1(x)
        if perturb:
            x1 = self.lsp_1(x1)
        self.l0 = x1
        x2 = self.conv2(x1)
        self.l1 = x2
        x3 = self.conv3(x2)
        self.l2 = x3
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        fea = x5.view(x5.size(0), -1)
        return fea, x3


class Classifier(nn.Module):
    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(nn.Linear(128 * 6, 128))
        self.out = nn.Linear(128, n_classes)

    def forward(self, x): return self.out(self.fc(x))


class Classifier_te(nn.Module):
    def __init__(self, n_classes):
        super(Classifier_te, self).__init__()
        self.fc = nn.Sequential(nn.Linear(128 * 6, 128))
        self.out = nn.Linear(128, n_classes)

    def forward(self, x): return self.out(self.fc(x))


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


def get_covariance_matrix(f_map, eye=None):
    eps = 1e-5
    B, C, H = f_map.shape
    HW = H
    if eye is None: eye = torch.eye(C).to(f_map.device)
    f_map = f_map.contiguous().view(B, C, -1)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW - 1) + (eps * eye)
    return f_cor, B


def get_cross_covariance_matrix(f_map1, f_map2, eye=None):
    eps = 1e-5
    B, C, H = f_map1.shape
    HW = H
    if eye is None: eye = torch.eye(C).to(f_map1.device)
    f_map1 = f_map1.contiguous().view(B, C, -1)
    f_map2 = f_map2.contiguous().view(B, C, -1)
    f_cor = torch.bmm(f_map1, f_map2.transpose(1, 2)).div(HW - 1) + (eps * eye)
    return f_cor, B


def cross_whitening_loss(k_feat, q_feat):
    f_cor, B = get_cross_covariance_matrix(k_feat, q_feat)
    diag_loss = torch.FloatTensor([0]).to(k_feat.device)
    for cor in f_cor:
        diag = torch.diagonal(cor.squeeze(dim=0), 0)
        eye = torch.ones_like(diag).to(k_feat.device)
        diag_loss = diag_loss + F.mse_loss(diag, eye)
    return diag_loss / B


def CORAL(source, target):
    d = source.shape[1]
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    return loss / (4 * d * d)


# ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()
    set_seed(42);
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(project="PU_Main_Benchmark", name=f"BDC_{args.source}", config=vars(args))
    train_loader = construct_loader('./data', 'PU', args.source, 40, True)
    test_loaders = [construct_loader('./data', 'PU', t, 40, False) for t in args.target.split(',')]

    CNN_st = CNN().to(device)
    CNN_te = CNN_Tea().to(device)
    classifier = Classifier(3).to(device)
    classifier_t = Classifier_te(3).to(device)

    criterion = nn.CrossEntropyLoss()
    Con = SupConLoss().to(device)

    lr = 0.0005
    weight_decay = 0.00005
    optimizer_ST = optim.Adam(CNN_st.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_TE = optim.Adam(CNN_te.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_LD = optim.Adam(CNN_te.lsp_1.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_classifier_t = optim.Adam(classifier_t.parameters(), lr=lr, weight_decay=weight_decay)

    tail_acc, tail_f1, tail_auc = [], [], []

    for epoch in range(1, 101):
        CNN_st.train();
        CNN_te.train();
        CNN_te.lsp_1.train()
        classifier.train();
        classifier_t.train()
        total_loss = 0.0

        for i, l in train_loader:
            s1_x, s1_y = i.to(device).float(), l.to(device).long()
            if len(s1_x.shape) == 2: s1_x = s1_x.unsqueeze(1)

            # --- 阶段一：一致性学习 ---
            optimizer_ST.zero_grad();
            optimizer_TE.zero_grad()
            optimizer_classifier.zero_grad();
            optimizer_classifier_t.zero_grad()

            features_st, st = CNN_st(s1_x)
            pre_st = classifier(features_st)
            loss_cls_st = criterion(pre_st, s1_y)

            features_te, te = CNN_te(s1_x, perturb=True)
            pre_te = classifier_t(features_te)
            loss_cls_te = criterion(pre_te, s1_y)

            st2_cm = get_covariance_matrix(st)[0]
            te2_cm = get_covariance_matrix(te)[0]
            consis_cml = F.l1_loss(st2_cm, te2_cm, reduction='mean')
            consis_ccl = cross_whitening_loss(st, te)

            emb_src = F.normalize(pre_st).unsqueeze(1)
            emb_aug = F.normalize(pre_te).unsqueeze(1)
            con = Con(torch.cat([emb_src, emb_aug], dim=1), s1_y)

            Loss_1 = loss_cls_st + loss_cls_te + 10 * consis_cml + 1 * consis_ccl + 0.1 * con

            # 【修复】：正常释放计算图，不允许 retain_graph
            Loss_1.backward()
            optimizer_ST.step();
            optimizer_classifier.step()
            optimizer_classifier_t.step();
            optimizer_TE.step()

            # --- 阶段二：差异化对抗学习 ---
            optimizer_LD.zero_grad()

            # 【致命细节修复】：必须进行第二次前向传播！重构新参数后的计算图
            _, _ = CNN_st(s1_x)
            _, _ = CNN_te(s1_x, perturb=True)

            l0 = CNN_te.l0
            l0_st = CNN_st.l0.detach()
            B = l0_st.size(0)

            coral_0 = CORAL(l0.view(B, -1), l0_st.view(B, -1))
            loss_2 = - 0.001 * coral_0

            loss_2.backward()
            optimizer_LD.step()

            total_loss += Loss_1.item()

        CNN_te.eval();
        classifier_t.eval()
        all_labels, all_preds, all_probs, all_features = [], [], [], []
        with torch.no_grad():
            for t_loader in test_loaders:
                for i, l in t_loader:
                    i = i.to(device).float();
                    i = i.unsqueeze(1) if len(i.shape) == 2 else i
                    features = CNN_te(i, perturb=False)[0]
                    logits = classifier_t(features)

                    all_features.extend(features.cpu().numpy())
                    all_labels.extend(l.numpy())
                    all_preds.extend(logits.argmax(1).cpu().numpy())
                    all_probs.extend(F.softmax(logits, dim=1).cpu().numpy())

        acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = 100. * f1_score(all_labels, all_preds, average='macro')
        auc = 100. * roc_auc_score(all_labels, all_probs, multi_class='ovr')

        if epoch > 80:
            tail_acc.append(acc);
            tail_f1.append(f1);
            tail_auc.append(auc)
        wandb.log({"Epoch": epoch, "Loss": total_loss / len(train_loader), "Acc": acc, "F1": f1, "AUC": auc})

        if epoch == 100:
            os.makedirs("plot_data", exist_ok=True)
            np.save(f"plot_data/features_BDC_{args.source}.npy", np.array(all_features))
            np.save(f"plot_data/labels_BDC_{args.source}.npy", np.array(all_labels))
            np.save(f"plot_data/preds_BDC_{args.source}.npy", np.array(all_preds))

    print(f"FINAL_ACCURACY:{np.mean(tail_acc):.2f}")
    wandb.log(
        {"Final_Avg_Acc": np.mean(tail_acc), "Final_Avg_F1": np.mean(tail_f1), "Final_Avg_AUC": np.mean(tail_auc)})
    wandb.finish()


if __name__ == '__main__':
    main()