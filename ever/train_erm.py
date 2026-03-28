import torch
import torch.optim as optim
import torch.nn as nn
from module import BDC_Net
import wandb
import os


def train_erm(train_loader, test_loader, args, device, log_path):
    # 初始化模型 (与 BDC 使用完全相同的架构)
    model = BDC_Net(num_classes=args.classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # 优化器设置 (保持与 BDC 一致)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    print(f">>> Start ERM Training")
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.long().to(device)
            # 维度调整: (B, 1024) -> (B, 1, 1024)
            if x.dim() == 2: x = x.unsqueeze(1)

            optimizer.zero_grad()

            # 【关键】BDC_Net forward 返回 6 个值: logits_o, logits_p, c3, p3, c1, p1
            # ERM 只需要原始路径的分类结果 logits_o (索引 0)
            logits_o = model(x)[0]

            loss = criterion(logits_o, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluation
        if (epoch + 1) % 1 == 0:
            acc = evaluate(model, test_loader, device)
            if acc > best_acc: best_acc = acc

            avg_loss = total_loss / len(train_loader)
            print(f"ERM Ep {epoch + 1:03d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} (Best: {best_acc:.4f})")

            # 本地日志
            if log_path:
                with open(log_path, 'a') as f:
                    # ERM 没有 Adv Loss，填 0 占位
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

            # 测试时也只取 logits_o
            logits = model(x)[0]
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0