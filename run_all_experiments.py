import subprocess
import pandas as pd
import sys

def run_cmd(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    final_acc = 0.0
    for line in process.stdout:
        # 捕获 WandB 的实时面板链接并打印
        if "View run at" in line:
            url = line.split("View run at")[-1].strip()
            print(f"    [🔗 Live Tracking] {url}")
        # 捕获每个模型跑完后的最终准确率
        elif "FINAL_ACCURACY:" in line:
            final_acc = float(line.split(":")[1])
    process.wait()
    return final_acc

def main():
    tasks = [
        ('N15_M01', 'N15_M07,N09_M07'),
        ('N15_M07', 'N15_M01,N09_M07'),
        ('N09_M07', 'N15_M01,N15_M07')
    ]

    # 终极五大核心 SOTA 矩阵
    models = {
        'ERM (50E)': 'main_erm.py',
        'DACN (50E)': 'main_dacn.py',
        'DWCN (100E)': 'main_DWCN.py',
        'BDC (100E)': 'main_bdc.py',
        'OURS (50E)': 'main.py'
    }

    print("=" * 70)
    print("🚀 Starting Final Benchmark Evaluation (5 Models)...")
    print("📊 Target Project: PU_Main_Benchmark")
    print("=" * 70)

    results = []
    for src, tgt in tasks:
        print(f"\nTask: {src} -> {tgt}")
        task_res = {'Source': src, 'Target': tgt}

        for name, script in models.items():
            print(f"  Evaluating {name:<20} ", flush=True)
            cmd = [sys.executable, script, "--source", src, "--target", tgt]
            acc = run_cmd(cmd)
            task_res[name] = acc
            print(f"  -> Score: {acc:.2f}%\n")

        results.append(task_res)
        pd.DataFrame(results).to_csv("Benchmark_Results_Temp.csv", index=False)

    df = pd.DataFrame(results)
    numeric_cols = df.columns.drop(['Source', 'Target'])
    df.loc['Average'] = df[numeric_cols].mean()
    df.at['Average', 'Source'] = 'ALL'
    df.at['Average', 'Target'] = 'ALL'

    df.to_csv("Benchmark_Results_Final.csv", index=False)
    print("\n✅ Final Evaluation complete. Results saved to Benchmark_Results_Final.csv.")
    print(df)

if __name__ == "__main__":
    main()