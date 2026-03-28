import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

# 设置论文画图的全局字体（如果需要宋体或Times New Roman，可自行修改）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14


def plot_confusion_matrix(y_true, y_pred, model_name, source_domain):
    """生成论文级混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5), dpi=300)
    # 使用 Blues 或 YlGnBu 这种学术常用的单色系渐变
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Class 0', 'Class 1', 'Class 2'],
                yticklabels=['Class 0', 'Class 1', 'Class 2'],
                cbar_kws={'label': 'Accuracy Rate'})

    plt.title(f'Confusion Matrix ({model_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    save_path = f"figures/CM_{model_name}_{source_domain}.pdf"
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存至: {save_path}")


def plot_tsne(features, labels, model_name, source_domain):
    """生成论文级 T-SNE 特征降维图"""
    print(f"正在计算 {model_name} 的 T-SNE (可能需要几十秒)...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(7, 6), dpi=300)

    # 论文常用的对比色调 (红、蓝、绿)
    colors = ['#d62728', '#1f77b4', '#2ca02c']
    markers = ['o', 's', '^']
    class_names = ['Class 0', 'Class 1', 'Class 2']

    for i in range(3):
        idx = (labels == i)
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1],
                    c=colors[i], marker=markers[i], label=class_names[i],
                    alpha=0.7, edgecolors='w', s=60)

    plt.title(f'T-SNE Feature Visualization ({model_name})')
    # 隐藏坐标轴刻度，让图看起来更高级
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='best')
    plt.tight_layout()

    save_path = f"figures/TSNE_{model_name}_{source_domain}.pdf"
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ T-SNE 图已保存至: {save_path}")


def main():
    # 假设我们想看 N15_M01 作为源域时，ERM 和 Fusion 的对比
    source_domain = "N15_M01"
    models_to_plot = ["ERM", "Fusion"]

    for model in models_to_plot:
        feat_path = f"plot_data/features_{model}_{source_domain}.npy"
        label_path = f"plot_data/labels_{model}_{source_domain}.npy"
        pred_path = f"plot_data/preds_{model}_{source_domain}.npy"

        if os.path.exists(feat_path) and os.path.exists(label_path):
            features = np.load(feat_path)
            labels = np.load(label_path)
            preds = np.load(pred_path)

            plot_confusion_matrix(labels, preds, model, source_domain)
            plot_tsne(features, labels, model, source_domain)
        else:
            print(f"⚠️ 找不到 {model} 的数据文件，请确保先运行了实验。")


if __name__ == "__main__":
    main()