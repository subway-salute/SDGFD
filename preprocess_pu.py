import os
import glob
import numpy as np
import scipy.io as scio

# ================= 配置区域 =================
# 严格匹配你的 tree 目录树：数据就在当前目录(code)下的 data 文件夹里
SRC_ROOT = r'./data/PU_raw'
DST_ROOT = r'./data/PU'

# 轴承分组 (保持不变)
BEARING_GROUPS = {
    0: ['K001', 'K002', 'K003', 'K004', 'K005'],
    1: ['KI04', 'KI14', 'KI16', 'KI18', 'KI21'],
    2: ['KA04', 'KA15', 'KA16', 'KA22', 'KA30']
}

WIN_LEN = 1024
STRIDE = 1024
# ==========================================

def load_mat_signal(fpath):
    try:
        mat = scio.loadmat(fpath)
        for key, val in mat.items():
            if key.startswith('__'): continue
            try:
                if 'Y' in val.dtype.names:
                    return val['Y'][0, 0]['Data'][0, 0].flatten()
            except:
                pass
            if isinstance(val, np.ndarray) and val.size > 50000:
                return val.flatten()
    except Exception as e:
        print(f"[Error] {fpath}: {e}")
    return None


def process_file(signal):
    if len(signal) < WIN_LEN: return None

    n_samples = (len(signal) - WIN_LEN) // STRIDE + 1
    samples = []

    for i in range(n_samples):
        # 1. 直接截取时域波形
        sig = signal[i * STRIDE: i * STRIDE + WIN_LEN]

        # 2. 纯时域 Z-Score 标准化 (消除基础绝对幅值差异，只保留振动形态比例)
        # 绝对不要做 FFT！把频域变换的工作留给 GPU 中的 MARS_Module 动态完成！
        std = np.std(sig)
        if std > 1e-6:
            sig = (sig - np.mean(sig)) / std
        else:
            sig = sig - np.mean(sig)

        samples.append(sig)

    return np.array(samples) if samples else None


if __name__ == '__main__':
    if not os.path.exists(DST_ROOT): os.makedirs(DST_ROOT)
    buffer = {}

    print(f"Scanning {SRC_ROOT} (Time-Domain Mode for SOTA Fusion)...")
    files = glob.glob(os.path.join(SRC_ROOT, "**/*.mat"), recursive=True)

    count = 0
    for fpath in files:
        fname = os.path.basename(fpath)

        label = None
        for cls, codes in BEARING_GROUPS.items():
            for c in codes:
                if c in fname:
                    label = cls
                    break
            if label is not None: break

        if label is None: continue

        # 提取工况，如 N15_M01
        parts = fname.split('_')
        cond = None
        for i in range(len(parts) - 1):
            if parts[i].startswith('N') and parts[i + 1].startswith('M'):
                cond = f"{parts[i]}_{parts[i + 1]}"
                break

        if not cond: continue

        if cond not in buffer: buffer[cond] = {'X': [], 'Y': []}

        raw = load_mat_signal(fpath)
        if raw is not None:
            samps = process_file(raw)
            if samps is not None:
                buffer[cond]['X'].append(samps)
                buffer[cond]['Y'].append(np.full(len(samps), label))
                count += 1
                if count % 100 == 0: print(f"Processed {count} files...")

    print("Saving...")
    for cond, data in buffer.items():
        if not data['X']: continue
        X = np.concatenate(data['X'], axis=0).astype(np.float32)
        # PyTorch 的 CrossEntropyLoss 要求 label 是 int64 (LongTensor)
        Y = np.concatenate(data['Y'], axis=0).astype(np.int64)

        save_dir = os.path.join(DST_ROOT, cond)
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        np.save(os.path.join(save_dir, 'data_X.npy'), X)
        np.save(os.path.join(save_dir, 'data_Y.npy'), Y)
        print(f"Saved {cond}: X shape {X.shape}, Y shape {Y.shape}")

    print("\n✅ 数据处理完毕！你可以直接去运行 main.py 了！")