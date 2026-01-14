import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class PUDataset(Dataset):
    def __init__(self, x_path, y_path, normalize=False):
        self.X = np.load(x_path).astype(np.float32)
        self.Y = np.load(y_path).astype(np.longlong)
        self.normalize = normalize

        print(f"Dataset loaded from {os.path.dirname(x_path)}. Shape: {self.X.shape}")
        if self.normalize:
            print(">> [Info] Z-Score Normalization Enabled (Sample-wise).")
        else:
            print(">> [Info] Using RAW Amplitude (No Normalization).")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        if self.normalize:
            # 学术标准: Sample-wise Z-Score
            # (x - mean) / std
            mean = np.mean(x)
            std = np.std(x)
            if std < 1e-8: std = 1e-8
            x = (x - mean) / std

        return x, y


def get_loader(source_path, batch_size, shuffle=True, normalize=False):
    x_path = os.path.join(source_path, 'data_X.npy')
    y_path = os.path.join(source_path, 'data_Y.npy')

    dataset = PUDataset(x_path, y_path, normalize=normalize)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
    return loader