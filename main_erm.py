import argparse, os, random, wandb, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from construct_loader import construct_loader
from module import CNN_Frontend

def set_seed(seed=42):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(project="PU_test2", name=f"ERM_{args.source}2{args.target}")
    train_loader = construct_loader('./data', 'PU', args.source, 40, True)
    test_loader = construct_loader('./data', 'PU', args.target, 40, False)

    model = nn.Sequential(CNN_Frontend(256), nn.Linear(256, 3)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    tail_acc, tail_f1, tail_auc = [], [], []

    for epoch in range(1, 41):
        model.train(); total_loss = 0.0
        for i, l in train_loader:
            i, l = i.to(device).float(), l.to(device)
            if len(i.shape) == 2: i = i.unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(i), l)
            loss.backward(); optimizer.step(); total_loss += loss.item()

        model.eval(); all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for i, l in test_loader:
                i = i.to(device).float()
                if len(i.shape) == 2: i = i.unsqueeze(1)
                logits = model(i)
                all_labels.extend(l.numpy())
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_probs.extend(F.softmax(logits, dim=1).cpu().numpy())

        acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        f1 = 100. * f1_score(all_labels, all_preds, average='macro')
        auc = 100. * roc_auc_score(all_labels, all_probs, multi_class='ovr')

        if epoch > 30:
            tail_acc.append(acc); tail_f1.append(f1); tail_auc.append(auc)

        wandb.log({"Epoch": epoch, "Loss": total_loss/len(train_loader), "Acc": acc, "F1": f1, "AUC": auc})

    print(f"ERM_FINAL_ACCURACY:{np.mean(tail_acc):.2f}")
    wandb.log({"Final_Avg_Acc": np.mean(tail_acc), "Final_Avg_F1": np.mean(tail_f1), "Final_Avg_AUC": np.mean(tail_auc)})
    wandb.finish()

if __name__ == '__main__':
    main()