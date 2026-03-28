import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class SConv_1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel, padding=1), nn.BatchNorm1d(out_ch), nn.ReLU())
    def forward(self, x): return self.conv(x)

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

class DWConv_Frontend(nn.Module):
    def __init__(self, end_feat=256):
        super().__init__()
        self.w1 = DWConv(num_channels=1); self.c1 = SConv_1D(2, 16, 3)
        self.w2 = DWConv(num_channels=16); self.c2 = SConv_1D(32, 64, 3)
        self.w3 = DWConv(num_channels=64); self.c3 = SConv_1D(128, end_feat, 3)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.c1(self.w1(x)); x = self.c2(self.w2(x)); x = self.c3(self.w3(x))
        return self.avg_pool(x).view(x.size(0), -1)

class CNN_Frontend(nn.Module):
    def __init__(self, end_feat=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 15, 2, 7), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, end_feat, 3, 1, 1), nn.BatchNorm1d(end_feat), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
    def forward(self, x): return self.net(x).view(x.size(0), -1)

class ResADAINGenerator(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.fc_gamma = nn.Linear(1, feature_dim)
        self.fc_beta = nn.Linear(1, feature_dim)

    def forward(self, x, z, y):
        mu, var = x.mean(dim=1, keepdim=True), x.var(dim=1, keepdim=True)
        x_normed = (x - mu.detach()) / (var + 1e-7).sqrt().detach()
        return x + (self.fc_gamma(z) * x_normed + self.fc_beta(y))

class DomainDiscriminator(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feature, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, 1)
        )
    def forward(self, x, alpha=1.0): return self.net(GRL.apply(x, alpha))

class LegoSOTANet(nn.Module):
    def __init__(self, frontend='wavelet', num_classes=3):
        super().__init__()
        self.fe = DWConv_Frontend(256) if frontend == 'wavelet' else CNN_Frontend(256)
        self.latent_gen = ResADAINGenerator(256)
        self.DI = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Linear(256, num_classes)
        self.projector = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 128))
        self.batchnorm_D = nn.BatchNorm1d(num_classes * 256)
        self.discriminator = DomainDiscriminator(num_classes * 256)

    def extract_FE(self, x): return self.fe(x)