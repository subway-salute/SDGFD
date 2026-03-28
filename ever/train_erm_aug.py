import torch
import torch.optim as optim
import torch.nn as nn
from module import BDC_Net
import wandb
import os


def train_erm_aug(train_loader, test_loader, args, device, log_path, noise_std=0.05):
    """
    ERM with Data Augmentation (Additive Gaussian Noise)
    noise_std: 噪声强度，默认 0.05
    """
    model = BDC_Net(num_classes=args.classes).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    print(f">>> Start ERM+Aug Training (Noise={noise_std})")
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.long().to(device)
            if x.dim() == 2: x = x.unsqueeze(1)

            # === Data Augmentation ===
            # 生成与输入形状一致的高斯噪声
            noise = torch.randn_like(x) * noise_std
            x_aug = x + noise
            # =========================

            optimizer.zero_grad()

            # 将增强后的数据传入模型，并只取 logits_o
            logits_o = model(x_aug)[0]

            loss = criterion(logits_o, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluation (测试时不加噪声)
        if (epoch + 1) % 1 == 0:
            acc = evaluate(model, test_loader, device)
            if acc > best_acc: best_acc = acc

            avg_loss = total_loss / len(train_loader)
            print(f"ERM_Aug Ep {epoch + 1:03d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} (Best: {best_acc:.4f})")

            # 本地日志
            if log_path:
                with open(log_path, 'a') as f:
                    f.write(f"{epoch + 1},{avg_loss:.4f},0.0000,{acc:.4f}\n")

            # WandB 上传
            wandb.log({
                "epoch": epoch + 1,
                "train/loss_task": avg_loss,
                "test/accuracy": acc,
                "test/best_acc": best_acc
            })


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.long().to(device)
            if x.dim() == 2: x = x.unsqueeze(1)

            # 测试只用原始路径，且不加噪声
            logits = model(x)[0]
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0