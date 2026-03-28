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


# ================== 模块 A: 前端提取器 ==================
class DWConv_Frontend(nn.Module):
    """DWCN 的小波物理前端"""

    def __init__(self, end_feat=256):
        super().__init__()
        # 此处省略小波基础类定义，使用你原有的 DWConv 和 SConv_1D 即可
        # 为保持代码简洁，这里直接用占位，请保留你原代码中的 DWConv 和 SConv_1D 定义
        self.w1 = DWConv(num_channels=1)
        self.c1 = SConv_1D(2, 16, 3)
        self.w2 = DWConv(num_channels=16)
        self.c2 = SConv_1D(32, 64, 3)
        self.w3 = DWConv(num_channels=64)
        self.c3 = SConv_1D(128, end_feat, 3)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.c1(self.w1(x))
        x = self.c2(self.w2(x))
        x = self.c3(self.w3(x))
        return self.avg_pool(x).view(x.size(0), -1)


class CNN_Frontend(nn.Module):
    """DACN 的纯数据驱动 CNN 前端 (Baseline 级别)"""

    def __init__(self, end_feat=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(64, end_feat, kernel_size=3, padding=1),
            nn.BatchNorm1d(end_feat), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        return self.net(x).view(x.size(0), -1)


# ================== 模块 B: 中端变异引擎 ==================
class ADAINGenerator(nn.Module):
    """DACN 狂野的纯 ADAIN 替换 (无残差保命)"""

    def __init__(self, noise_dim=1, feature_dim=256):
        super().__init__()
        self.fc_gamma = nn.Linear(noise_dim, feature_dim)
        self.fc_beta = nn.Linear(noise_dim, feature_dim)

    def forward(self, x, z, y):
        mu, var = x.mean(dim=1, keepdim=True), x.var(dim=1, keepdim=True)
        x_normed = (x - mu.detach()) / (var + 0.1).sqrt().detach()
        return self.fc_gamma(z) * x_normed + self.fc_beta(y)


class ResADAINGenerator(nn.Module):
    """DWCN 保守的残差 ADAIN"""

    def __init__(self, noise_dim=1, feature_dim=256):
        super().__init__()
        self.fc_gamma = nn.Linear(noise_dim, feature_dim)
        self.fc_beta = nn.Linear(noise_dim, feature_dim)

    def forward(self, x, z, y):
        mu, var = x.mean(dim=1, keepdim=True), x.var(dim=1, keepdim=True)
        x_normed = (x - mu.detach()) / (var + 0.1).sqrt().detach()
        return x + (self.fc_gamma(z) * x_normed + self.fc_beta(y))


# ================== 后端组件 ==================
class DomainDiscriminator(nn.Module):
    def __init__(self, in_feature, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feature, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(hidden_size, 1)
        )

    def forward(self, x, alpha=1.0):
        return self.net(GRL.apply(x, alpha))


# ================== 终极动态装配车间 ==================
class LegoSOTANet(nn.Module):
    def __init__(self, num_classes=3, end_feat=256, frontend='wavelet', mutation='res_adain', adv='cdan'):
        super().__init__()
        # 1. 组装前端
        self.fe = DWConv_Frontend(end_feat) if frontend == 'wavelet' else CNN_Frontend(end_feat)

        # 2. 组装变异器
        self.latent_gen = ResADAINGenerator(1, end_feat) if mutation == 'res_adain' else ADAINGenerator(1, end_feat)

        # 3. 公共处理模块
        self.DI = nn.Sequential(nn.Linear(end_feat, end_feat), nn.BatchNorm1d(end_feat), nn.ReLU(inplace=True),
                                nn.Dropout(0.3))
        self.projector = nn.Sequential(nn.Linear(end_feat, end_feat), nn.BatchNorm1d(end_feat), nn.ReLU(inplace=True),
                                       nn.Linear(end_feat, end_feat // 2))
        self.classifier = nn.Linear(end_feat, num_classes)

        # 4. 组装对抗网络
        self.adv_type = adv
        if adv == 'cdan':
            self.batchnorm_D = nn.BatchNorm1d(num_classes * end_feat)
            self.discriminator = DomainDiscriminator(num_classes * end_feat)
        elif adv == 'dann':
            self.discriminator = DomainDiscriminator(end_feat)

    def extract_FE(self, x):
        return self.fe(x)