import os
import torch
import argparse
import numpy as np
from load_data import load_data_npy
from construct_loader import construct_loader
from train_bdc import train_bdc
from train_erm import train_erm
from train_erm_aug import train_erm_aug


def main():
    parser = argparse.ArgumentParser()
    # 模式选择: bdc, erm, erm_aug
    parser.add_argument('--mode', type=str, default='bdc', help='bdc / erm / erm_aug')
    parser.add_argument('--source_path', type=str, required=True)
    parser.add_argument('--target_paths', type=str, nargs='+', required=True)

    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.00005)
    parser.add_argument('--class_sample', type=int, default=0)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # === 日志设置 ===
    if not os.path.exists('result'):
        os.makedirs('result')

    source_name = os.path.basename(args.source_path)
    # 文件名加时间戳或后缀防止覆盖，这里简单用 mode+source
    log_path = os.path.join('result', f"log_{args.mode}_{source_name}.txt")

    with open(log_path, 'w') as f:
        f.write("Epoch,Loss_Task,Loss_Adv,Test_Acc\n")

    print(f"\nMode: {args.mode.upper()} | Source: {source_name}")
    print(f"Logging to: {log_path}\n")

    # === 加载数据 ===
    sx = os.path.join(args.source_path, 'data_X.npy')
    sy = os.path.join(args.source_path, 'data_Y.npy')
    X_s, Y_s = load_data_npy(sx, sy, class_sample=args.class_sample)
    if X_s.ndim == 2: X_s = X_s[:, np.newaxis, :]
    train_loader = construct_loader(X_s, Y_s, args.batch_size, shuffle=True)

    X_t_list, Y_t_list = [], []
    for t_path in args.target_paths:
        tx = os.path.join(t_path, 'data_X.npy')
        ty = os.path.join(t_path, 'data_Y.npy')
        xt, yt = load_data_npy(tx, ty)
        if xt.ndim == 2: xt = xt[:, np.newaxis, :]
        X_t_list.append(xt)
        Y_t_list.append(yt)

    X_t = np.concatenate(X_t_list, axis=0)
    Y_t = np.concatenate(Y_t_list, axis=0)
    test_loader = construct_loader(X_t, Y_t, args.batch_size, shuffle=False)

    # ... (前面的代码)
    X_t = np.concatenate(X_t_list, axis=0)
    Y_t = np.concatenate(Y_t_list, axis=0)

    # ======== 【新增】 打印数据指纹，验证到底加载了谁 ========
    print(f"\n[DEBUG CHECK] 正在检查测试集数据源...")
    print(f"  > 测试集样本数: {X_t.shape[0]}")
    print(f"  > 测试集数据均值 (Mean): {X_t.mean():.6f}")
    print(f"  > 测试集数据方差 (Std):  {X_t.std():.6f}")
    print(f"  > 目标域路径列表: {args.target_paths}")

    # 对比一下源域数据
    print(f"[DEBUG CHECK] 对比源域数据...")
    print(f"  > 源域样本数: {X_s.shape[0]}")
    print(f"  > 源域数据均值 (Mean): {X_s.mean():.6f}")
    # =======================================================

    test_loader = construct_loader(X_t, Y_t, args.batch_size, shuffle=False)

    # === 训练路由 ===
    if args.mode == 'bdc':
        train_bdc(train_loader, test_loader, args, device, log_path)
    elif args.mode == 'erm':
        train_erm(train_loader, test_loader, args, device, log_path)
    elif args.mode == 'erm_aug':
        train_erm_aug(train_loader, test_loader, args, device, log_path, noise_std=0.05)


if __name__ == '__main__':
    main()