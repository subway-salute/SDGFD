import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from module import BDC_Net, CORAL, get_covariance_matrix, cross_whitening_loss, SupConLoss
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

# === 参数设置 (保持官方参数，但 lambda_adv 恢复官方 -0.001) ===
# 注意：既然你的 CORAL 已经有了除数，且 FFT 也对齐了，
# 我们应该尝试回归官方的 lambda_adv = -0.001。
# 如果发现 Adv 依然为 0 (下溢)，可以微调为 -0.01 或 -0.1，但绝不是 -10.0 (那是给无除数版用的)
CFG = {
    'lr': 0.0005,
    'weight_decay': 0.00005,
    'epochs': 150,
    'lambda_cm': 10.0,
    'lambda_pc': 1.0,
    'lambda_sc': 0.1,
    'lambda_adv': -0.001,  # 尝试回归官方值，或者 -0.01
    'grad_clip': 1.0
}


def train_bdc(train_loader, test_loader, args, device, log_path):
    model = BDC_Net(num_classes=args.classes).to(device)
    criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss()

    # === 优化器定义 (关键修正) ===
    # 官方 main.py:
    # optimizer_TE = optim.Adam(CNN_te.parameters()...) -> 包含 LSP
    # optimizer_LD = optim.Adam(CNN_te.lsp_1.parameters()...) -> 只包含 LSP

    # 对应到这里：
    # opt_all: 更新整个网络 (对应 optimizer_TE + optimizer_classifier)
    opt_all = optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])

    # opt_lsp: 只更新 LSP (对应 optimizer_LD)
    lsp_params = [p for n, p in model.named_parameters() if 'lsp' in n]
    opt_lsp = optim.Adam(lsp_params, lr=CFG['lr'], weight_decay=CFG['weight_decay'])

    print(f">>> Start BDC (Corrected Training Strategy)")

    history = {'acc': [], 'f1': [], 'rec': []}

    for epoch in range(CFG['epochs']):
        model.train()
        meter_task = 0.0
        meter_adv = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.long().to(device)
            if x.dim() == 2: x = x.unsqueeze(1)

            # ====================================================
            # Phase 1: Task & Consistency (对应官方 Loss_1)
            # 目标：分类准确 + 特征一致 (Constraint)
            # 关键点：LSP 在这里也要更新！不能冻结！
            # ====================================================

            # 1. 清空所有梯度
            opt_all.zero_grad()

            # 2. Forward
            # BDC_Net 内部逻辑:
            # c系列: clean path (source)
            # p系列: perturbed path (target style) -> 经过了 lsp
            logits_o, logits_p, c3, p3, c1, p1 = model(x)

            # 3. Loss 1 Calculation
            loss_cls = criterion(logits_o, y) + criterion(logits_p, y)

            cov_o = get_covariance_matrix(c3)
            cov_p = get_covariance_matrix(p3)
            loss_cm = F.l1_loss(cov_o, cov_p)

            loss_pc = cross_whitening_loss(c3, p3)

            emb_o = F.normalize(logits_o, dim=1).unsqueeze(1)
            emb_p = F.normalize(logits_p, dim=1).unsqueeze(1)
            loss_sc = con_criterion(torch.cat([emb_o, emb_p], dim=1), y)

            loss_1 = loss_cls + CFG['lambda_cm'] * loss_cm + CFG['lambda_pc'] * loss_pc + CFG['lambda_sc'] * loss_sc

            # 4. Update All (包括 LSP!)
            loss_1.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
            opt_all.step()

            meter_task += loss_cls.item()

            # ====================================================
            # Phase 2: Discrepancy (对应官方 Loss_2)
            # 目标：最大化差异 (Adversarial)
            # 关键点：只更新 LSP
            # ====================================================

            opt_lsp.zero_grad()  # 只清空 LSP 的梯度 (虽然 opt_all 也清空了，但保险起见)

            # 这里需要重新 forward 吗？官方代码似乎是在同一个 batch 里做了两次 forward
            # 或者是利用计算图保留。为了节省显存，通常重新 forward。
            # 但要注意 detach CNN 的部分。

            # 官方逻辑回顾：
            # optimizer_LD.zero_grad()
            # features_st, _ = CNN_st(s1_x)
            # features_te, _ = CNN_te(s1_x, perturb=True)
            # ...
            # loss_2.backward()
            # optimizer_LD.step()

            # 我们这里为了防止更新 CNN 参数，只让 LSP require grad 是没用的，因为 opt_lsp 只包含 LSP 参数
            # 但为了计算图不回传到 CNN 前层，最好 detach 输入给 LSP 的特征，或者依靠 opt_lsp 的参数列表限制。
            # 在 PyTorch 中，如果 optimizer 不包含某参数，该参数即便有梯度也不会更新。
            # 所以直接 forward 即可。

            logits_o, logits_p, c3, p3, c1, p1 = model(x)

            c1_flat = c1.view(c1.size(0), -1)
            p1_flat = p1.view(p1.size(0), -1)

            # Clean vs Perturbed
            loss_coral = CORAL(c1_flat, p1_flat)

            # Maximize Discrepancy = Minimize (lambda * CORAL) where lambda < 0
            loss_lsp = CFG['lambda_adv'] * loss_coral

            loss_lsp.backward()
            torch.nn.utils.clip_grad_norm_(lsp_params, CFG['grad_clip'])
            opt_lsp.step()

            meter_adv += loss_lsp.item()

        # Evaluation
        if (epoch + 1) % 1 == 0:
            acc, f1, rec = evaluate(model, test_loader, device)

            history['acc'].append(acc)
            history['f1'].append(f1)
            history['rec'].append(rec)

            last_n = 10
            avg_acc = np.mean(history['acc'][-last_n:])

            avg_task = meter_task / len(train_loader)
            avg_adv = meter_adv / len(train_loader)

            print(f"BDC Ep {epoch + 1:03d} | Task: {avg_task:.4f} | Adv: {avg_adv:.2e} | "
                  f"Cur Acc: {acc:.4f} | Avg Acc: {avg_acc:.4f}")

            if log_path:
                with open(log_path, 'a') as f:
                    f.write(f"{epoch + 1},{avg_task:.4f},{avg_adv:.6f},{acc:.4f},{f1:.4f},{rec:.4f}\n")

            wandb.log({
                "epoch": epoch + 1,
                "train/loss_task": avg_task,
                "train/loss_adv": avg_adv,
                "test/accuracy": acc,
                "test/f1_score": f1,
                "test/recall": rec,
                "test/last10_avg_acc": avg_acc
            })

    print(f"\n>>> BDC Finished. Final Last-10 Avg Acc: {np.mean(history['acc'][-10:]):.4f}")


def evaluate(model, loader, device):
    model.eval()
    preds_list = []
    labels_list = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.long().to(device)
            if x.dim() == 2: x = x.unsqueeze(1)
            # 官方测试逻辑：perturb=False，即只用 clean path
            logits, _, _, _, _, _ = model(x)
            pred = logits.argmax(dim=1)
            preds_list.append(pred.cpu().numpy())
            labels_list.append(y.cpu().numpy())

    all_preds = np.concatenate(preds_list)
    all_labels = np.concatenate(labels_list)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    return acc, f1, rec