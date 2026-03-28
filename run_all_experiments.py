import subprocess
import pandas as pd
import sys


def run_cmd(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    final_acc = 0.0
    for line in process.stdout:
        print(line, end='')  # 实时打印，让你知道它没有卡死
        if "FINAL_ACCURACY:" in line:
            final_acc = float(line.split(":")[1])
    process.wait()
    return final_acc


def main():
    # 三种不同的源域与目标域组合
    tasks = [
        ('N15_M01', 'N09_M07'),
        ('N15_M01', 'N15_M07'),
        ('N15_M07', 'N09_M07')
    ]

    models = {
        'ERM': 'main_erm.py',
        'DWCN': 'main_DWCN.py',
        'DACN': 'main_dacn.py',
        'OURS(Fusion)': 'main.py'
    }

    results = []

    for src, tgt in tasks:
        print(f"\n" + "=" * 50)
        print(f"🚀 开始测试组合: {src} -> {tgt}")
        print("=" * 50)
        task_res = {'Source': src, 'Target': tgt}

        for name, script in models.items():
            print(f"\n--- 正在运行 {name} ---")
            cmd = [sys.executable, script, "--source", src, "--target", tgt]
            acc = run_cmd(cmd)
            task_res[name] = acc
            print(f"Result for {name}: {acc}%")

        results.append(task_res)
        pd.DataFrame(results).to_csv("Benchmark_Results_Temp.csv", index=False)

    df = pd.DataFrame(results)
    df.to_csv("Benchmark_Results_Final.csv", index=False)
    print("\n✅ 所有对比实验已完成！结果已保存至 Benchmark_Results_Final.csv")
    print(df)


if __name__ == "__main__":
    main()