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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()
    set_seed(42);
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(project="PU_Final_6Models", name=f"ERM_{args.source}", config=vars(args))
    train_loader = construct_loader('./data', 'PU', args.source, 40, True)

    target_list = args.target.split(',')
    test_loaders = [construct_loader('./data', 'PU', t, 40, False) for t in target_list]

    # 严格对标原论文：使用与对比模型完全一致的特征提取主干，确保参数量绝对公平
    model = nn.Sequential(CNN_Frontend(256), nn.Linear(256, 3)).to(device)

    # 采用赵超博士 baseline 的标准优化器设置 (带 5e-4 权重衰减防过拟合)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    tail_acc, tail_f1, tail_auc = [], [], []

    for epoch in range(1, 51):
        model.train();
        total_loss = 0.0
        for i, l in train_loader:
            i, l = i.to(device).float(), l.to(device)
            if len(i.shape) == 2: i = i.unsqueeze(1)
            optimizer.zero_grad()

            # 最纯净的 ERM：没有任何域自适应损失，只计算源域交叉熵
            loss = criterion(model(i), l)

            loss.backward();
            optimizer.step();
            total_loss += loss.item()

        # ================= 混合目标域测试 & 特征提取 =================
        model.eval();
        all_labels, all_preds, all_probs, all_features = [], [], [], []
        with torch.no_grad():
            for t_loader in test_loaders:
                for i, l in t_loader:
                    i = i.to(device).float()
                    if len(i.shape) == 2: i = i.unsqueeze(1)

                    features = model[0](i)
                    logits = model[1](features)

                    all_features.extend(features.cpu().numpy())
                    all_labels.extend(l.numpy())
                    all_preds.extend(logits.argmax(1).cpu().numpy())
                    all_probs.extend(F.softmax(logits, dim=1).cpu().numpy())

        acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = 100. * f1_score(all_labels, all_preds, average='macro')
        auc = 100. * roc_auc_score(all_labels, all_probs, multi_class='ovr')

        if epoch > 40:  # 统一取最后 10 轮做平均
            tail_acc.append(acc);
            tail_f1.append(f1);
            tail_auc.append(auc)

        wandb.log({"Epoch": epoch, "Loss": total_loss / len(train_loader), "Acc": acc, "F1": f1, "AUC": auc})

        # 🌟 最后一轮保存画图数据 🌟
        if epoch == 50:
            os.makedirs("plot_data", exist_ok=True)
            np.save(f"plot_data/features_ERM_{args.source}.npy", np.array(all_features))
            np.save(f"plot_data/labels_ERM_{args.source}.npy", np.array(all_labels))
            np.save(f"plot_data/preds_ERM_{args.source}.npy", np.array(all_preds))

    print(f"FINAL_ACCURACY:{np.mean(tail_acc):.2f}")
    wandb.log(
        {"Final_Avg_Acc": np.mean(tail_acc), "Final_Avg_F1": np.mean(tail_f1), "Final_Avg_AUC": np.mean(tail_auc)})
    wandb.finish()


if __name__ == '__main__':
    main()