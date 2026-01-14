import torch
import torch.utils.data as Data
import numpy as np


def construct_loader(x, y, batch_size, shuffle=True):
    """
    构建 DataLoader，兼容 Numpy 和 Tensor 输入
    """
    # 1. 检查并转换 X (Feature)
    if not isinstance(x, torch.Tensor):
        x = torch.FloatTensor(x)

    # 2. 检查并转换 Y (Label) -> 必须是 LongTensor 用于 CrossEntropy
    if not isinstance(y, torch.Tensor):
        # 如果是 numpy，转为 Tensor
        y = torch.LongTensor(y)
    else:
        # 如果已经是 Tensor，确保是 long 类型
        y = y.long()

    dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader