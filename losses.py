import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # 修正了归一化维度，彻底防止多视角特征扭曲
        features = F.normalize(features, dim=-1)
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = features.shape[1]

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask.repeat(anchor_count, anchor_count) * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        # 工业级防爆补丁：+ 1e-9 彻底杜绝 log(0) 引发的 NaN
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1.0, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()