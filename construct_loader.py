import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader


# ================= 定义 Worker 初始化函数 =================
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ===============================================================


class BearingDataset(Dataset):
    def __init__(self, data_root, dataset_name, domain):
        # 【核心修改点】：直接拼接，对应真实路径 ./data/PU/N15_M01
        domain_path = os.path.join(data_root, dataset_name, domain)

        x_path = os.path.join(domain_path, "data_X.npy")
        y_path = os.path.join(domain_path, "data_Y.npy")

        if not os.path.exists(x_path):
            raise FileNotFoundError(f"Missing data at {x_path}")

        self.X = np.load(x_path)
        self.Y = np.load(y_path)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).view(1, -1)
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        return x, y


def construct_loader(data_root, dataset_name, domain, batch_size, is_train=True):
    dataset = BearingDataset(data_root, dataset_name, domain)

    # ================= 创建被绝对锁死的生成器 =================
    g = torch.Generator()
    g.manual_seed(42)
    # ===============================================================

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g
    )