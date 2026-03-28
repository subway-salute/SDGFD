import numpy as np
import torch
from construct_loader import construct_loader
import os


def load_data(source_path, target_path, batch_size):
    # ================= 修改开始：适配 .npy 数据 =================
    print(f"Loading Source (NPY): {source_path}")
    print(f"Loading Target (NPY): {target_path}")

    # 1. 读取 Source
    s_x = np.load(os.path.join(source_path, 'data_X.npy'))
    s_y = np.load(os.path.join(source_path, 'data_Y.npy'))

    # 2. 读取 Target (你的 target_paths 可能是 list，这里做兼容)
    if isinstance(target_path, list):
        # 如果是 list，合并所有 target 数据
        t_x_list, t_y_list = [], []
        for p in target_path:
            t_x_list.append(np.load(os.path.join(p, 'data_X.npy')))
            t_y_list.append(np.load(os.path.join(p, 'data_Y.npy')))
        t_x = np.concatenate(t_x_list, axis=0)
        t_y = np.concatenate(t_y_list, axis=0)
    else:
        # 单个路径
        t_x = np.load(os.path.join(target_path, 'data_X.npy'))
        t_y = np.load(os.path.join(target_path, 'data_Y.npy'))

    # 3. 维度调整 (N, 1024) -> (N, 1, 1024)
    # 官方代码期望 (Batch, 1, Length)
    if s_x.ndim == 2: s_x = s_x[:, np.newaxis, :]
    if t_x.ndim == 2: t_x = t_x[:, np.newaxis, :]

    # 4. 转 Tensor
    source_x = torch.from_numpy(s_x).float()
    source_y = torch.from_numpy(s_y).long()
    target_x = torch.from_numpy(t_x).float()
    target_y = torch.from_numpy(t_y).long()
    # ================= 修改结束 =================

    # 官方原版只构造了 Train_loader
    # Test_loader 是我们在 main 里手动用的，所以这里 return Tensor 也是可以的
    # 但根据官方 construct_loader.py，我们需要返回 list

    loader = construct_loader([source_x, source_y], batch_size)

    # 官方 main.py 直接用了 load_data 返回的 source_x, source_y (Tensor)
    # 仔细看官方 main.py:
    # data = load_data(...) -> data 是一个 loader
    # 咦？官方 main.py 里写的是 Train_loader = construct_loader(...)
    # 抱歉，我应该看官方 main.py 是怎么调用的。

    # 修正：官方 main.py 里并没有 import load_data!
    # 官方 main.py 是把加载逻辑写死在里面的 (Process data ...)。
    # 既然你要我“在原来的代码基础上修改”，那我就把数据加载逻辑直接写进 main.py，
    # 就像官方那样。

    # 但为了清晰，这里我们保留这个文件作为工具函数供 main.py 调用
    return source_x, source_y, target_x, target_y