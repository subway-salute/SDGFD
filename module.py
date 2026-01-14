import torch
import torch.nn as nn
import torch.nn.functional as F


class LSP(nn.Module):
    """
    论文核心模块: Learnable Statistical Perturbation (Eq. 10)
    位置: 第1、2层卷积之后
    """

    def __init__(self, num_channels):
        super(LSP, self).__init__()
        # [修改点] 初始化不再是全0，而是微小的随机噪声，打破对称性
        self.f_mu = nn.Parameter(torch.randn(1, num_channels, 1) * 0.01)
        self.f_sigma = nn.Parameter(torch.randn(1, num_channels, 1) * 0.01)

    def forward(self, x):
        # x: (Batch, Channel, Length)
        mu = x.mean(dim=2, keepdim=True)
        sigma = x.std(dim=2, keepdim=True) + 1e-6

        # Instance Norm
        x_norm = (x - mu) / sigma

        # [修改点] 确保 sigma 系数不会导致数值反转太剧烈，限制范围
        # 使用 softplus 确保方差扰动因子大体为正，或者直接相加但监控数值
        # 这里保持论文原意直接相加，但 relying on initialization
        return (sigma + self.f_sigma) * x_norm + (mu + self.f_mu)


class BDC_Net(nn.Module):
    """
    论文 IV-C: 5-layer CNN + 2-layer FC Classifier
    """

    def __init__(self, num_classes=3):
        super(BDC_Net, self).__init__()

        # === Layer 1 ===
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lsp1 = LSP(64)

        # === Layer 2 ===
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lsp2 = LSP(64)

        # === Layer 3 ===
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # === Layer 4 ===
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # === Layer 5 ===
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )

        # Classifier Input: 4096 (128 channels * 32 length)
        self.classifier = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # --- Path 1: Original ---
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3_o = self.conv3(c2)
        c4 = self.conv4(c3_o)
        c5 = self.conv5(c4)
        logits_o = self.classifier(c5)
        feat_o = self.classifier[0](c5)
        proj_o = self.head(feat_o)

        # --- Path 2: Perturbed ---
        p1 = self.lsp1(c1)
        p2 = self.conv2(p1)
        p2 = self.lsp2(p2)
        c3_p = self.conv3(p2)
        p4 = self.conv4(c3_p)
        p5 = self.conv5(p4)
        logits_p = self.classifier(p5)
        feat_p = self.classifier[0](p5)
        proj_p = self.head(feat_p)

        return logits_o, logits_p, c3_o, c3_p, proj_o, proj_p