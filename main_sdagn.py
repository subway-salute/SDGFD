import argparse, os, random, wandb, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from construct_loader import construct_loader
from module import CNN_Frontend


def set_seed(seed=42):
    random.seed(seed);
    os.environ['PYTHONHASHSEED'] = str(seed);
    np.random.seed(seed)
    torch.manual_seed(seed);
    torch.cuda.manual_seed(seed);
    torch.backends.cudnn.deterministic = True


# ==================== 严格照搬 SDAGN 官方 utils.py ====================
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def pdist_torch(emb1, emb2):
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(emb1, emb2.t(), beta=1, alpha=-2)
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


class BatchHardTripletSelector(object):
    # 绝对尊重原作者，保留 numpy 的运算逻辑防止变形
    def __call__(self, embeds, labels):
        dist_mtx = pdist_torch(embeds, embeds).detach().cpu().numpy()
        labels = labels.contiguous().cpu().numpy().reshape((-1, 1))
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)
        lb_eqs = labels == labels.T
        lb_eqs[dia_inds] = False
        dist_same = dist_mtx.copy()
        dist_same[lb_eqs == False] = -np.inf
        pos_idxs = np.argmax(dist_same, axis=1)
        dist_diff = dist_mtx.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        neg_idxs = np.argmin(dist_diff, axis=1)
        pos = embeds[pos_idxs].contiguous().view(num, -1)
        neg = embeds[neg_idxs].contiguous().view(num, -1)
        return embeds, pos, neg


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        return self.Loss(anchor, pos, neg)


# ====================================================================

# ================= 严格复刻 SDAGN 官方的输入级 Mixup =================
def mix_aug_random(src_data, cls_label):
    aug_data = []
    aug_label = []
    classes = torch.unique(cls_label)

    for c in classes:
        c_data = src_data[cls_label == c]
        n = c_data.size(0)
        if n < 2: continue
        # 按照原厂逻辑，对每个类生成等量的混合新样本
        for _ in range(n):
            a = random.random()
            b = 1.0 - a
            indices = torch.randperm(n)[:2]
            aug_data.append(a * c_data[indices[0]] + b * c_data[indices[1]])
            aug_label.append(c)

    if len(aug_data) > 0:
        return torch.stack(aug_data), torch.stack(aug_label)
    return None, None


# ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()
    set_seed(42);
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(project="PU_Thesis_Final", name=f"SDAGN_{args.source}", config=vars(args))
    train_loader = construct_loader('./data', 'PU', args.source, 40, True)
    test_loaders = [construct_loader('./data', 'PU', t, 40, False) for t in args.target.split(',')]

    model = nn.Sequential(CNN_Frontend(256), nn.Linear(256, 3)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion_cls = nn.CrossEntropyLoss()
    triplet_loss_fn = TripletLoss(margin=1.0).to(device)
    selector = BatchHardTripletSelector()

    tail_acc, tail_f1, tail_auc = [], [], []

    for epoch in range(1, 41):
        model.train();
        total_loss = 0.0
        for i, l in train_loader:
            i, l = i.to(device).float(), l.to(device)
            if len(i.shape) == 2: i = i.unsqueeze(1)
            optimizer.zero_grad()

            # 1. 提取源域特征与预测
            src_features = model[0](i)
            src_logits = model[1](src_features)

            # 2. 生成 Mixup 数据并提取特征
            aug_data, aug_label = mix_aug_random(i, l)
            if aug_data is not None:
                aug_features = model[0](aug_data)
                aug_logits = model[1](aug_features)

                # ================= 核心计算: 语义 CMMD Loss =================
                sematic_loss = 0.0
                classes = torch.unique(l)
                for c in classes:
                    src_f_c = src_features[l == c]
                    aug_f_c = aug_features[aug_label == c]
                    if src_f_c.size(0) > 0 and aug_f_c.size(0) > 0:
                        min_sz = min(src_f_c.size(0), aug_f_c.size(0))
                        sematic_loss += mmd_rbf_noaccelerate(src_f_c[:min_sz], aug_f_c[:min_sz])

                # ================= 核心计算: 困难三元组 Loss =================
                feats = torch.cat((src_features, aug_features), dim=0)
                one_label = torch.cat((l, aug_label), dim=0)

                anchor, pos, neg = selector(feats, one_label)
                triplet = triplet_loss_fn(anchor, pos, neg)

                # 恢复完全体 SDAGN：a=1.0, b=1.0
                cls_loss = criterion_cls(src_logits, l) + criterion_cls(aug_logits, aug_label)
                loss = cls_loss + 1.0 * sematic_loss + 1.0 * triplet
            else:
                loss = criterion_cls(src_logits, l)

            loss.backward();
            optimizer.step();
            total_loss += loss.item()

        # ================= 测试与特征留存 =================
        model.eval();
        all_labels, all_preds, all_probs, all_features = [], [], [], []
        with torch.no_grad():
            for t_loader in test_loaders:
                for i, l in t_loader:
                    i = i.to(device).float();
                    i = i.unsqueeze(1) if len(i.shape) == 2 else i
                    features = model[0](i)
                    logits = model[1](features)

                    all_features.extend(features.cpu().numpy())
                    all_labels.extend(l.numpy())
                    all_preds.extend(logits.argmax(1).cpu().numpy())
                    all_probs.extend(F.softmax(logits, dim=1).cpu().numpy())

        acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = 100. * f1_score(all_labels, all_preds, average='macro')
        auc = 100. * roc_auc_score(all_labels, all_probs, multi_class='ovr')

        if epoch > 30: tail_acc.append(acc); tail_f1.append(f1); tail_auc.append(auc)
        wandb.log({"Epoch": epoch, "Loss": total_loss / len(train_loader), "Acc": acc, "F1": f1, "AUC": auc})

        if epoch == 40:
            os.makedirs("plot_data", exist_ok=True)
            np.save(f"plot_data/features_SDAGN_{args.source}.npy", np.array(all_features))
            np.save(f"plot_data/labels_SDAGN_{args.source}.npy", np.array(all_labels))
            np.save(f"plot_data/preds_SDAGN_{args.source}.npy", np.array(all_preds))

    print(f"FINAL_ACCURACY:{np.mean(tail_acc):.2f}")
    wandb.log(
        {"Final_Avg_Acc": np.mean(tail_acc), "Final_Avg_F1": np.mean(tail_f1), "Final_Avg_AUC": np.mean(tail_auc)})
    wandb.finish()


if __name__ == '__main__':
    main()