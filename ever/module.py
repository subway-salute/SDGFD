import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ================= 1. 损失函数与工具 =================

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


def get_covariance_matrix(f_map):
    eps = 1e-5
    B, C, H = f_map.shape
    f_map = f_map.contiguous().view(B, C, -1)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(H - 1) + (eps * torch.eye(C).to(f_map.device))
    return f_cor


def cross_whitening_loss(k_feat, q_feat):
    B, C, H = k_feat.shape
    f_map1 = k_feat.contiguous().view(B, C, -1)
    f_map2 = q_feat.contiguous().view(B, C, -1)
    f_cor = torch.bmm(f_map1, f_map2.transpose(1, 2)).div(H - 1)

    diag_loss = 0
    eye = torch.ones(C).to(k_feat.device)
    for i in range(B):
        diag = torch.diagonal(f_cor[i])
        diag_loss += F.mse_loss(diag, eye)
    return diag_loss / B


def CORAL(source, target):
    d = source.shape[1]
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt
    loss = torch.mean((xc - xct) ** 2)
    loss = loss / (4 * d * d)
    return loss


# ================= 2. 网络架构 =================

class LSP(nn.Module):
    def __init__(self, in_features):
        super(LSP, self).__init__()
        self.fc1 = nn.Linear(in_features, 1)
        self.fc2 = nn.Linear(in_features, 1)

    def forward(self, x):
        h1 = self.fc1(x)
        h2 = self.fc2(x)
        return (1 + h1) * x + h2


class BDC_Net(nn.Module):
    def __init__(self, num_classes=3):
        super(BDC_Net, self).__init__()

        # Layer 1: 官方 Kernel=32
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, 32, stride=1),
            nn.InstanceNorm1d(16, affine=False),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lsp1 = LSP(496)

        # Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, 3),
            nn.InstanceNorm1d(32, affine=False),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, 3),
            nn.InstanceNorm1d(64, affine=False),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )

        # Layer 4
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 128, 3),
            nn.InstanceNorm1d(128, affine=False),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )

        # Layer 5
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, 3),
            nn.InstanceNorm1d(128, affine=False),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self._feat_dim = 128 * 6

        self.classifier = nn.Sequential(
            nn.Linear(self._feat_dim, 128),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)

        # Path 1: Source (Clean)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        flat_o = c5.view(c5.size(0), -1)
        logits_o = self.classifier(flat_o)

        # Path 2: Perturbed (LSP)
        p1 = self.conv1(x)
        p1 = self.lsp1(p1)
        p2 = self.conv2(p1)
        p3 = self.conv3(p2)
        p4 = self.conv4(p3)
        p5 = self.conv5(p4)
        flat_p = p5.view(p5.size(0), -1)
        logits_p = self.classifier(flat_p)

        return logits_o, logits_p, c3, p3, c1, p1