import torch
import torch.optim as optim
import torch.nn as nn
from module import BDC_Net


def train_erm_aug(train_loader, test_loader, args, device, log_path, noise_std=0.05):
    # 使用 BDC_Net 保持骨干一致
    model = BDC_Net(num_classes=args.classes).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    print(f">>> Start ERM+Aug Training (Noise={noise_std})...")
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.long().to(device)
            if x.dim() == 2: x = x.unsqueeze(1)

            # === Data Augmentation ===
            noise = torch.randn_like(x) * noise_std
            x_aug = x + noise
            # =========================

            optimizer.zero_grad()
            # BDC_Net 前向传播，只取 logits_o
            logits_o, _, _, _, _, _ = model(x_aug)

            loss = criterion(logits_o, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluation
        if (epoch + 1) % 1 == 0:
            acc = evaluate(model, test_loader, device)
            if acc > best_acc: best_acc = acc

            avg_loss = total_loss / len(train_loader)
            print(f"ERM_Aug Ep {epoch + 1:03d} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f}")

            # 写入文件
            if log_path:
                with open(log_path, 'a') as f:
                    f.write(f"{epoch + 1},{avg_loss:.4f},0.0000,{acc:.4f}\n")

    print(f"ERM_Aug Finished. Best Accuracy: {best_acc:.4f}")


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.long().to(device)
            if x.dim() == 2: x = x.unsqueeze(1)
            logits, _, _, _, _, _ = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0