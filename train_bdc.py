import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from module import BDC_Net

# ================= 论文原配置 (现在可以放心用了) =================
CFG = {
    'lr': 0.0005,
    'epochs': 300,
    'lambda_cm': 10.0,
    'lambda_pc': 1.0,
    'lambda_sc': 0.1,
    'lambda_adv': -1.0
}


# =================================================

def compute_losses(logits_o, logits_p, f_o, f_p, proj_o, proj_p, labels):
    """计算所有 Loss 组件 (修复了分母计算 bug)"""
    criterion = nn.CrossEntropyLoss()
    l_task = criterion(logits_o, labels) + criterion(logits_p, labels)

    # f_o shape: (Batch, Channel, Length)
    b, c, l = f_o.size()

    # 展平: (Batch*Length, Channel)
    # 这意味着我们将 (Batch * Length) 视为总样本数 N
    f_o_flat = f_o.permute(0, 2, 1).reshape(-1, c)
    f_p_flat = f_p.permute(0, 2, 1).reshape(-1, c)

    # 样本总数 N
    N = f_o_flat.size(0)  # = b * l

    # Centering
    f_o_centered = f_o_flat - f_o_flat.mean(dim=0, keepdim=True)
    f_p_centered = f_p_flat - f_p_flat.mean(dim=0, keepdim=True)

    # === [关键修复] ===
    # 之前除以 b (40)，现在除以 N (5120)
    # 这样协方差矩阵的数值量级才会正常
    cov_o = torch.matmul(f_o_centered.t(), f_o_centered) / (N - 1)
    cov_p = torch.matmul(f_p_centered.t(), f_p_centered) / (N - 1)

    l_cm = torch.mean(torch.abs(cov_o - cov_p))

    # PC Loss
    f_o_std = f_o_flat.std(dim=0, keepdim=True) + 1e-6
    f_p_std = f_p_flat.std(dim=0, keepdim=True) + 1e-6
    f_o_norm = f_o_centered / f_o_std
    f_p_norm = f_p_centered / f_p_std

    # 同样除以 N-1
    cross_cov = torch.mm(f_o_norm.t(), f_p_norm) / (N - 1)
    target = torch.eye(c, device=f_o.device)
    l_pc = torch.norm(cross_cov - target, p='fro')

    l_sc = F.mse_loss(proj_o, proj_p)

    return l_task, l_cm, l_pc, l_sc


def train_bdc(train_loader, test_loader, args, device, log_path):
    model = BDC_Net(num_classes=args.classes).to(device)

    cnn_params = []
    lsp_params = []
    for name, param in model.named_parameters():
        if 'lsp' in name:
            lsp_params.append(param)
        else:
            cnn_params.append(param)

    opt_cnn = optim.Adam(cnn_params, lr=CFG['lr'])
    opt_lsp = optim.Adam(lsp_params, lr=CFG['lr'])

    print(f">>> Start BDC Strict Training (Epochs: {CFG['epochs']})")
    print(f">>> Config: {CFG}")

    best_acc = 0.0

    for epoch in range(CFG['epochs']):
        model.train()
        loss_task_sum = 0
        loss_adv_sum = 0

        debug_cm = 0
        debug_pc = 0

        for x, y in train_loader:
            x, y = x.to(device), y.long().to(device)
            if x.dim() == 2: x = x.unsqueeze(1)

            # === Phase 1: CNN Update ===
            for p in lsp_params: p.requires_grad = False
            for p in cnn_params: p.requires_grad = True

            logits_o, logits_p, f_o, f_p, proj_o, proj_p = model(x)
            l_task, l_cm, l_pc, l_sc = compute_losses(logits_o, logits_p, f_o, f_p, proj_o, proj_p, y)

            # 现在 Loss 量级正常了，直接相加即可
            loss_1 = l_task + CFG['lambda_cm'] * l_cm + CFG['lambda_pc'] * l_pc + CFG['lambda_sc'] * l_sc

            opt_cnn.zero_grad()
            loss_1.backward()
            opt_cnn.step()

            loss_task_sum += l_task.item()
            debug_cm += l_cm.item()
            debug_pc += l_pc.item()

            # === Phase 2: LSP Update ===
            for p in lsp_params: p.requires_grad = True
            for p in cnn_params: p.requires_grad = False

            logits_o, logits_p, f_o, f_p, proj_o, proj_p = model(x)
            _, l_cm_val, _, _ = compute_losses(logits_o, logits_p, f_o, f_p, proj_o, proj_p, y)

            loss_2 = CFG['lambda_adv'] * l_cm_val

            opt_lsp.zero_grad()
            loss_2.backward()
            opt_lsp.step()
            loss_adv_sum += loss_2.item()

        # Evaluation
        if (epoch + 1) % 1 == 0:
            acc = evaluate(model, test_loader, device)
            if acc > best_acc: best_acc = acc

            # 打印 Mean Loss以便观察量级
            avg_task = loss_task_sum / len(train_loader)
            avg_adv = loss_adv_sum / len(train_loader)
            avg_cm = debug_cm / len(train_loader)
            avg_pc = debug_pc / len(train_loader)

            print(f"Ep {epoch + 1:03d} | Task: {avg_task:.2f} | CM: {avg_cm:.2e} | PC: {avg_pc:.1f} | Acc: {acc:.4f}")

            if log_path:
                with open(log_path, 'a') as f:
                    f.write(f"{epoch + 1},{loss_task_sum:.4f},{loss_adv_sum:.4f},{acc:.4f}\n")

    print(f"Training Finished. Best Accuracy: {best_acc:.4f}")


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