import os
import scipy.io as scio
import numpy as np
from scipy.fftpack import fft

# ================= 配置区域 =================
# 原始数据路径 (你的raw文件夹)
SRC_ROOT = './data/cwru/raw'
# 处理后保存路径
DST_ROOT = './data/cwru/cwru_processed'

# CWRU 文件名映射表 (12k Drive End 数据)
# 格式: {工况名: {类别名: [文件名列表]}}
# 假设你下载的文件名如下 (97.mat, 105.mat ...)，请确保raw文件夹里有这些文件
CWRU_MAP = {
    '0HP': {
        'Normal': ['97.mat'], 'Inner': ['105.mat'], 'Ball': ['118.mat'], 'Outer': ['130.mat']
    },
    '1HP': {
        'Normal': ['98.mat'], 'Inner': ['106.mat'], 'Ball': ['119.mat'], 'Outer': ['131.mat']
    },
    '2HP': {
        'Normal': ['99.mat'], 'Inner': ['107.mat'], 'Ball': ['120.mat'], 'Outer': ['132.mat']
    },
    '3HP': {
        'Normal': ['100.mat'], 'Inner': ['108.mat'], 'Ball': ['121.mat'], 'Outer': ['133.mat']
    }
}

LABELS = {'Normal': 0, 'Inner': 1, 'Ball': 2, 'Outer': 3}


# ===========================================

# 打开 preprocess_pu.py，找到 preprocess_signal 函数，修改如下：

def preprocess_signal(signal):
    if len(signal) < 2048: return None
    signal = signal[:2048]
    f_val = fft(signal)

    # [修改前] f_val = 2 * np.abs(f_val) / 2048
    # [修改后] 严格遵守论文 "no normalization"
    f_val = np.abs(f_val)

    f_val = f_val[:1024]
    return f_val


def generate_samples(data, stride=2048):
    samples = []
    n_samples = (len(data) - 2048) // stride + 1
    for i in range(n_samples):
        sig = data[i * stride: i * stride + 2048]
        proc = preprocess_signal(sig)
        if proc is not None:
            samples.append(proc)
    return np.array(samples)


def process_cwru():
    if not os.path.exists(SRC_ROOT):
        print(f"Error: Source directory {SRC_ROOT} not found.")
        return

    print(f"Start processing CWRU data from {SRC_ROOT}...")

    for load_name, class_map in CWRU_MAP.items():
        save_dir = os.path.join(DST_ROOT, load_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        all_X, all_Y = [], []

        for cls_name, files in class_map.items():
            label = LABELS[cls_name]
            for fname in files:
                fpath = os.path.join(SRC_ROOT, fname)
                if not os.path.exists(fpath):
                    print(f"  [Warn] File {fname} not found in {load_name}, skipping.")
                    continue

                try:
                    mat = scio.loadmat(fpath)
                    # 自动寻找 DE_time key
                    key = next(k for k in mat.keys() if 'DE_time' in k)
                    raw_data = mat[key].flatten()

                    # 生成样本
                    samps = generate_samples(raw_data)
                    if len(samps) > 0:
                        all_X.append(samps)
                        all_Y.append(np.full(len(samps), label))
                        print(f"  Load {load_name} | {cls_name} ({fname}): {len(samps)} samples.")
                except Exception as e:
                    print(f"  [Error] Failed to process {fname}: {e}")

        if all_X:
            X = np.concatenate(all_X, axis=0)
            Y = np.concatenate(all_Y, axis=0)
            np.save(os.path.join(save_dir, 'data_X.npy'), X)
            np.save(os.path.join(save_dir, 'data_Y.npy'), Y)
            print(f"Saved {load_name} to {save_dir} | Shape: {X.shape}")
        else:
            print(f"No data found for {load_name}.")


if __name__ == '__main__':
    process_cwru()