import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.5):  # 🌟 已修改为 0.5 降温防爆
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features shape: [batch_size, n_views, projection_dim] (例如 [40, 2, 128])
        features = F.normalize(features, dim=-1)
        device = features.device
        batch_size = features.shape[0]

        # 1. 基础 mask 生成 (大小: batch_size x batch_size，例如 40x40)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 2. 特征展开 (将源域和变异域特征在 batch 维度拼接)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = features.shape[1]  # 提取视角数量，这里是 2

        # 3. 计算相似度 logits (大小: 80x80)
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        # 数值稳定补丁：减去最大值防指数爆炸
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 4. 🚀 修复后的 Mask 扩展与对角线屏蔽逻辑
        # (此时 anchor_count 和 mask 都已正确定义)
        mask = mask.repeat(anchor_count, anchor_count)  # 先把 40x40 铺成 80x80

        # 生成一个 80x80 的矩阵，对角线为 0，其他为 1
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0
        )
        # 去除自己和自己的正样本匹配
        mask = mask * logits_mask

        # 5. 计算 Loss (带 1e-9 终极防爆补丁)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1.0, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - mean_log_prob_pos
        return loss.view(anchor_count, batch_size).mean()