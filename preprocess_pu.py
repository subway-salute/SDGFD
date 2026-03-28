import os
import glob
import numpy as np
import scipy.io as scio

# ================= 配置区域 =================
SRC_ROOT = r'./data/PU_raw'
DST_ROOT = r'./data/PU'

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
        sig = signal[i * STRIDE: i * STRIDE + WIN_LEN]
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
                if c in fname: label = cls; break
            if label is not None: break
        if label is None: continue

        parts = fname.split('_')
        cond = None
        for i in range(len(parts) - 1):
            if parts[i].startswith('N') and parts[i + 1].startswith('M'):
                cond = f"{parts[i]}_{parts[i + 1]}";
                break
        if not cond: continue

        if cond not in buffer: buffer[cond] = {'X': [], 'Y': []}

        raw = load_mat_signal(fpath)
        if raw is not None:
            samps = process_file(raw)
            if samps is not None:
                # 暂时将所有切片扁平化存入列表
                for s in samps:
                    buffer[cond]['X'].append(s)
                    buffer[cond]['Y'].append(label)
                count += 1
                if count % 100 == 0: print(f"Processed {count} files...")

    print("\n================ 开始执行严格的类别对齐 (Class Balancing) ================")
    for cond, data in buffer.items():
        if not data['X']: continue

        y_array = np.array(data['Y'])
        classes = np.unique(y_array)

        # 统计每个类别的索引
        class_indices = {c: np.where(y_array == c)[0] for c in classes}

        # 找到样本数最少的类别数量
        min_count = min([len(idx) for idx in class_indices.values()])
        print(
            f"[{cond}] 原始样本分布: { {c: len(idx) for c, idx in class_indices.items()} } -> 强制对齐至每个类: {min_count} 个")

        balanced_X, balanced_Y = [], []

        # 对每一个类别，随机抽取 min_count 个样本，保证绝对公平
        for c, idx in class_indices.items():
            # 锁定随机种子保证每次切分的数据一致
            np.random.seed(42)
            selected_idx = np.random.choice(idx, min_count, replace=False)
            for i in selected_idx:
                balanced_X.append(data['X'][i])
                balanced_Y.append(data['Y'][i])

        X = np.stack(balanced_X).astype(np.float32)
        Y = np.array(balanced_Y).astype(np.int64)

        save_dir = os.path.join(DST_ROOT, cond)
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        np.save(os.path.join(save_dir, 'data_X.npy'), X)
        np.save(os.path.join(save_dir, 'data_Y.npy'), Y)
        print(f"✅ Saved {cond}: X shape {X.shape}, Y shape {Y.shape}")

    print("\n🎉 数据处理及对齐完毕！你可以直接去运行 main.py 了！")