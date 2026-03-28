import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


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


class SConv_1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.conv(x)


class ResidualLatentGenerator(nn.Module):
    def __init__(self, noise_dim=1, feature_dim=256):
        super().__init__()
        self.fc_gamma = nn.Linear(noise_dim, feature_dim)
        self.fc_beta = nn.Linear(noise_dim, feature_dim)

    def forward(self, x, z, y):
        # DWCN 实例级标准化
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        x_normed = (x - mu.detach()) / (var + 0.1).sqrt().detach()

        gamma = self.fc_gamma(z)
        beta = self.fc_beta(y)

        # 核心防坍塌设计：残差连接
        return x + (gamma * x_normed + beta)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_feature, hidden_size=256, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feature, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, alpha=1.0):
        x = GRL.apply(x, alpha)
        return self.net(x)


class FusionSOTANet(nn.Module):
    def __init__(self, num_classes=3, end_feat=256):
        super().__init__()
        # 1. DWCN 小波降噪特征提取器 (3 层深化至 256 维)
        self.w1 = DWConv(num_channels=1)
        self.c1 = SConv_1D(2, 16, 3)
        self.w2 = DWConv(num_channels=16)
        self.c2 = SConv_1D(32, 64, 3)
        self.w3 = DWConv(num_channels=64)
        self.c3 = SConv_1D(128, end_feat, 3)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # 2. 安全残差扰动器
        self.latent_gen = ResidualLatentGenerator(noise_dim=1, feature_dim=end_feat)

        # 3. 域独立特征提取 (DI)
        self.DI = nn.Sequential(
            nn.Linear(end_feat, end_feat),
            nn.BatchNorm1d(end_feat),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # 4. 对比学习投影头
        self.projector = nn.Sequential(
            nn.Linear(end_feat, end_feat),
            nn.BatchNorm1d(end_feat),
            nn.ReLU(inplace=True),
            nn.Linear(end_feat, end_feat // 2)
        )

        # 5. 分类器
        self.classifier = nn.Linear(end_feat, num_classes)

        # 6. DACN 原厂级防爆对抗网络
        self.batchnorm_D = nn.BatchNorm1d(num_classes * end_feat)
        self.discriminator = DomainDiscriminator(in_feature=end_feat * num_classes)

    def extract_FE(self, x):
        x = self.c1(self.w1(x))
        x = self.c2(self.w2(x))
        x = self.c3(self.w3(x))
        return self.avg_pool(x).view(x.size(0), -1)