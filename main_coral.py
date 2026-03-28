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


def coral_loss(source, target):
    d = source.size(1)
    # 计算协方差矩阵
    xm = torch.mean(source, 0, keepdim=True)
    xc = source - xm
    source_c = torch.matmul(torch.transpose(xc, 0, 1), xc) / (source.size(0) - 1)

    tm = torch.mean(target, 0, keepdim=True)
    tc = target - tm
    target_c = torch.matmul(torch.transpose(tc, 0, 1), tc) / (target.size(0) - 1)

    # Frobenius norm
    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c))) / (4 * d * d)
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()
    set_seed(42);
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(project="PU_Thesis_Final", name=f"CORAL_{args.source}", config=vars(args))
    train_loader = construct_loader('./data', 'PU', args.source, 40, True)
    test_loaders = [construct_loader('./data', 'PU', t, 40, False) for t in args.target.split(',')]

    fe = CNN_Frontend(256).to(device)
    cls = nn.Linear(256, 3).to(device)
    optimizer = torch.optim.Adam(list(fe.parameters()) + list(cls.parameters()), lr=0.001)
    criterion_cls = nn.CrossEntropyLoss()

    tail_acc, tail_f1, tail_auc = [], [], []

    for epoch in range(1, 41):
        fe.train();
        cls.train();
        total_loss = 0.0
        for i, l in train_loader:
            i, l = i.to(device).float(), l.to(device)
            if len(i.shape) == 2: i = i.unsqueeze(1)
            optimizer.zero_grad()

            i_noisy = i + torch.randn_like(i) * 0.5

            f_src = fe(i)
            f_tgt = fe(i_noisy)

            loss_cls = criterion_cls(cls(f_src), l)
            loss_cr = coral_loss(f_src, f_tgt)

            loss = loss_cls + 1.0 * loss_cr
            loss.backward();
            optimizer.step();
            total_loss += loss.item()

        fe.eval();
        cls.eval();
        all_labels, all_preds, all_probs, all_features = [], [], [], []
        with torch.no_grad():
            for t_loader in test_loaders:
                for i, l in t_loader:
                    i = i.to(device).float();
                    i = i.unsqueeze(1) if len(i.shape) == 2 else i
                    features = fe(i)
                    logits = cls(features)

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
            np.save(f"plot_data/features_CORAL_{args.source}.npy", np.array(all_features))
            np.save(f"plot_data/labels_CORAL_{args.source}.npy", np.array(all_labels))
            np.save(f"plot_data/preds_CORAL_{args.source}.npy", np.array(all_preds))

    print(f"FINAL_ACCURACY:{np.mean(tail_acc):.2f}")
    wandb.log(
        {"Final_Avg_Acc": np.mean(tail_acc), "Final_Avg_F1": np.mean(tail_f1), "Final_Avg_AUC": np.mean(tail_auc)})
    wandb.finish()


if __name__ == '__main__':
    main()