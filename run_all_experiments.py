import os
import subprocess
import time

def main():
    domains = ["N09_M07", "N15_M01", "N15_M07"]
    data_root = "./DATA"
    dataset = "PU"

    # 新架构（一致性正则化）下的推荐超参数
    mix_p = 0.24
    cons_weight = 1.0  # 已将 domain_weight 替换为一致性约束权重 cons_weight
    dropout = 0.3
    epochs = 100       # 抛弃 DANN 后，100 轮绝对安全且收敛更好
    num_classes = 3

    print("-" * 60)
    print("🚀 Starting cross-domain evaluation (Consistency Reg. Architecture)")
    print(f"Dataset: {dataset} | Classes: {num_classes}")
    print(f"Parameters: mix_p={mix_p}, cons_weight={cons_weight}, dropout={dropout}, epochs={epochs}")
    print("-" * 60)

    # 3 个域两两组合，共 6 个跨域任务。每个任务跑 Baseline 和 Proposed，总计 12 次训练。
    total_tasks = len(domains) * (len(domains) - 1) * 2
    current_task = 1

    for source in domains:
        for target in domains:
            if source == target:
                continue

            print(f"\nTask [{current_task}/{total_tasks}]: Source: {source} -> Target: {target}")

            # ================= 1. Baseline =================
            print(f"  -> Training Baseline model...")
            cmd_baseline = (
                f"python main.py --data_root {data_root} --dataset {dataset} "
                f"--source {source} --target {target} --num_classes {num_classes} "
                f"--no_mix --no_white --dropout {dropout} --epoch {epochs} --seed 42"
            )
            start_time = time.time()
            subprocess.run(cmd_baseline, shell=True)
            print(f"  -> Baseline finished. Time elapsed: {(time.time() - start_time) / 60:.2f} min.")
            current_task += 1

            # ================= 2. Proposed (Cons. Reg.) =================
            print(f"  -> Training Proposed model (Consistency Reg.)...")
            cmd_proposed = (
                f"python main.py --data_root {data_root} --dataset {dataset} "
                f"--source {source} --target {target} --num_classes {num_classes} "
                f"--mix_p {mix_p} --cons_weight {cons_weight} --dropout {dropout} --epoch {epochs} --seed 42"
            )
            start_time = time.time()
            subprocess.run(cmd_proposed, shell=True)
            print(f"  -> Proposed finished. Time elapsed: {(time.time() - start_time) / 60:.2f} min.")
            current_task += 1

    print("\n🎉 All experimental tasks completed. Please check WandB [PU_Consistency_Test] for results.")

if __name__ == "__main__":
    main()