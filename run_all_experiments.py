import subprocess
import pandas as pd
import sys


def run_cmd(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    final_acc = 0.0
    for line in process.stdout:
        print(line, end='')
        if "FINAL_ACCURACY:" in line:
            final_acc = float(line.split(":")[1])
    process.wait()
    return final_acc


def main():
    # 绝对平衡数据的留一法混合测试
    tasks = [
        ('N15_M01', 'N15_M07,N09_M07'),
        ('N15_M07', 'N15_M01,N09_M07'),
        ('N09_M07', 'N15_M01,N15_M07')
    ]

    # 6大核心模型满编矩阵 (含两大2024年顶刊)
    models = {
        'ERM': 'main_erm.py',
        'CORAL': 'main_coral.py',
        'DANN': 'main_dann.py',
        'DWCN': 'main_DWCN.py',
        'SDAGN': 'main_sdagn.py',
        'OURS(Fusion)': 'main.py'
    }

    results = []

    for src, tgt in tasks:
        print(f"\n" + "=" * 75)
        print(f"🚀 [完全体对决] 留一法测试 源域: {src} -> 混合目标域: {tgt}")
        print("=" * 75)
        task_res = {'Source': src, 'Target': tgt}

        for name, script in models.items():
            print(f"\n--- 正在全速运行 {name} ---")
            cmd = [sys.executable, script, "--source", src, "--target", tgt]
            acc = run_cmd(cmd)
            task_res[name] = acc
            print(f"✅ {name} 最终得分: {acc:.2f}%")

        results.append(task_res)
        pd.DataFrame(results).to_csv("Benchmark_Results_Temp.csv", index=False)

    df = pd.DataFrame(results)

    # 计算所有任务的平均分，一锤定音
    numeric_cols = df.columns.drop(['Source', 'Target'])
    df.loc['Average'] = df[numeric_cols].mean()
    df.at['Average', 'Source'] = 'ALL'
    df.at['Average', 'Target'] = 'ALL'

    df.to_csv("Benchmark_Results_Final.csv", index=False)
    print("\n🎉 毕业设计所有对比实验已完美收官！请查看带有 Average 的大表 Benchmark_Results_Final.csv")
    print(df)


if __name__ == "__main__":
    main()