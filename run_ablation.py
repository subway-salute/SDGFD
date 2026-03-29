import subprocess
import pandas as pd
import sys


def run_cmd(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    final_acc = 0.0
    for line in process.stdout:
        if "View run at" in line:
            url = line.split("View run at")[-1].strip()
            print(f"    [🔗 Live Tracking] {url}")
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

    variants = {
        'Var A (Basic CNN)': 'A',
        'Var B (+ Wavelet)': 'B',
        'Var C (+ Wavelet + ADAIN)': 'C',
        'Var D (Full Model)': 'D'
    }

    print("=" * 70)
    print("🧩 Starting Ablation Study...")
    print("📊 Target Project: PU_Ablation_Study")
    print("=" * 70)

    results = []
    for src, tgt in tasks:
        print(f"\nTask: {src} -> {tgt}")
        task_res = {'Source': src, 'Target': tgt}

        for name, var_code in variants.items():
            print(f"  Evaluating {name:<25} ", flush=True)
            cmd = [sys.executable, "main_ablation.py", "--source", src, "--target", tgt, "--variant", var_code]
            acc = run_cmd(cmd)
            task_res[name] = acc
            print(f"  -> Score: {acc:.2f}%\n")

        results.append(task_res)
        pd.DataFrame(results).to_csv("Ablation_Results_Temp.csv", index=False)

    df = pd.DataFrame(results)
    numeric_cols = df.columns.drop(['Source', 'Target'])
    df.loc['Average'] = df[numeric_cols].mean()
    df.at['Average', 'Source'] = 'ALL'
    df.at['Average', 'Target'] = 'ALL'

    df.to_csv("Ablation_Results_Final.csv", index=False)
    print("\n✅ Ablation Study complete. Results saved to Ablation_Results_Final.csv.")
    print(df)


if __name__ == "__main__":
    main()