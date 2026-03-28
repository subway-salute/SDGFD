import os
import torch
import argparse
import numpy as np
import random
import wandb

from load_data import load_data_npy
from construct_loader import construct_loader
from train_bdc import train_bdc
from train_erm import train_erm
from train_erm_aug import train_erm_aug


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run_training(mode, args, train_loader, test_loader, device, source_name):
    print(f"\n{'=' * 40}")
    print(f"Starting Training Mode: {mode.upper()}")
    print(f"{'=' * 40}")

    # 日志文件
    log_path = os.path.join('../result', f"log_{mode}_{source_name}.txt")
    with open(log_path, 'w') as f:
        f.write("Epoch,Loss_Task,Loss_Adv,Test_Acc\n")

    # WandB Init (分组管理)
    wandb.init(
        project="PU",
        name=f"{mode}_{source_name}",
        group="Comparison_Experiment",  # 分组，方便对比
        config=vars(args),
        reinit=True
    )

    # 训练路由
    if mode == 'bdc':
        train_bdc(train_loader, test_loader, args, device, log_path)
    elif mode == 'erm':
        train_erm(train_loader, test_loader, args, device, log_path)
    elif mode == 'erm_aug':
        train_erm_aug(train_loader, test_loader, args, device, log_path, noise_std=0.05)

    wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    # 支持 'all' 模式
    parser.add_argument('--mode', type=str, default='all', choices=['bdc', 'erm', 'erm_aug', 'all'])
    parser.add_argument('--source_path', type=str, required=True)
    parser.add_argument('--target_paths', type=str, nargs='+', required=True)

    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.00005)  # 官方值
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--class_sample', type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists('../result'): os.makedirs('../result')

    # === 加载数据 (只加载一次) ===
    source_name = os.path.basename(args.source_path)
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

    # === 执行循环 ===
    modes_to_run = ['erm', 'bdc', 'erm_aug'] if args.mode == 'all' else [args.mode]

    for mode in modes_to_run:
        run_training(mode, args, train_loader, test_loader, device, source_name)


if __name__ == '__main__':
    main()