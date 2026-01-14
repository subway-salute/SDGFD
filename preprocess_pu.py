import os
import glob
import numpy as np
import scipy.io as scio
from scipy.fftpack import fft

# ================= 配置区域 =================
SRC_ROOT = r'data/PU_raw'  # 原始 .mat 文件夹路径
DST_ROOT = r'data/PU/PU_processed'  # 预处理输出路径

# 学术界标准分组 (Based on Table 5 & Table 7 of PU Dataset)
BEARING_GROUPS = {
    # Label 0: 健康 (混合 K001-K005)
    0: ['K001', 'K002', 'K003', 'K004', 'K005'],

    # Label 1: 真实内圈损伤 (混合 5 颗)
    1: ['KI04', 'KI14', 'KI16', 'KI18', 'KI21'],

    # Label 2: 真实外圈损伤 (混合 5 颗)
    2: ['KA04', 'KA15', 'KA16', 'KA22', 'KA30']
}

WIN_LEN = 2048  # 滑窗长度
STRIDE = 1024  # 步长 (50% 重叠)
FFT_LEN = 1024  # 特征长度


def load_mat_signal(fpath):
    """鲁棒读取 PU 数据集深层嵌套的 struct"""
    try:
        mat = scio.loadmat(fpath)
        for key, val in mat.items():
            if key.startswith('__'): continue
            # 路径: struct -> Y -> Data
            try:
                if 'Y' in val.dtype.names:
                    data = val['Y'][0, 0]['Data'][0, 0]
                    return data.flatten()
            except:
                pass
            # 兜底: 找最长的数组
            if isinstance(val, np.ndarray) and val.size > 100000:
                return val.flatten()
    except Exception as e:
        print(f"Error reading {fpath}: {e}")
    return None


def process_file(signal):
    """切片 + FFT (注意：这里不做归一化，保留原始幅值)"""
    n_samples = (len(signal) - WIN_LEN) // STRIDE + 1
    samples = []
    for i in range(n_samples):
        sig = signal[i * STRIDE: i * STRIDE + WIN_LEN]

        # FFT 变换
        f_val = fft(sig)
        f_val = np.abs(f_val)
        f_val = f_val[:FFT_LEN]
        samples.append(f_val)

    return np.array(samples) if samples else None


if __name__ == '__main__':
    if not os.path.exists(DST_ROOT): os.makedirs(DST_ROOT)

    # 用于按工况(N15_M07等)汇总数据
    buffer = {}

    print(f"Scanning {SRC_ROOT}...")
    # 递归搜索所有 .mat 文件
    files = glob.glob(os.path.join(SRC_ROOT, "**/*.mat"), recursive=True)

    count = 0
    for fpath in files:
        fname = os.path.basename(fpath)

        # 1. 匹配类别
        label = None
        for cls, codes in BEARING_GROUPS.items():
            for c in codes:
                if c in fname:
                    label = cls
                    break
            if label is not None: break

        if label is None: continue  # 跳过不在列表里的文件

        # 2. 解析工况 (Nxx_Mxx)
        parts = fname.split('_')
        cond = None
        for i in range(len(parts) - 1):
            if parts[i].startswith('N') and parts[i + 1].startswith('M'):
                cond = f"{parts[i]}_{parts[i + 1]}"
                break

        if not cond: continue

        # 3. 处理
        if cond not in buffer: buffer[cond] = {'X': [], 'Y': []}

        raw_sig = load_mat_signal(fpath)
        if raw_sig is not None:
            samps = process_file(raw_sig)
            if samps is not None:
                buffer[cond]['X'].append(samps)
                buffer[cond]['Y'].append(np.full(len(samps), label))
                count += 1
                if count % 20 == 0: print(f"Processed {count} files... ({fname})")

    # 4. 保存
    print("\nSaving data...")
    for cond, data in buffer.items():
        if not data['X']: continue
        X = np.concatenate(data['X'], axis=0)
        Y = np.concatenate(data['Y'], axis=0)

        out_dir = os.path.join(DST_ROOT, cond)
        if not os.path.exists(out_dir): os.makedirs(out_dir)

        np.save(os.path.join(out_dir, 'data_X.npy'), X)
        np.save(os.path.join(out_dir, 'data_Y.npy'), Y)
        print(f"Condition [{cond}]: {X.shape} samples saved.")