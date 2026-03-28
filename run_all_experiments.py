import itertools
import subprocess
import pandas as pd
import time
import sys


def main():
    print("🚀 开始穷举架构搜索... 任务：最难工况 N15_M01 -> N09_M07")

    # 定义搜索空间
    frontends = ['cnn', 'wavelet']
    mutations = ['adain', 'res_adain']
    aligns = ['none', 'supcon']
    advs = ['none', 'dann', 'cdan']

    all_combinations = list(itertools.product(frontends, mutations, aligns, advs))
    results = []

    start_time = time.time()

    for idx, (f, m, al, ad) in enumerate(all_combinations):
        print("\n" + "=" * 60)
        print(f"🔄 [{idx + 1}/{len(all_combinations)}] 当前组合: 前端={f} | 变异={m} | 聚类={al} | 对抗={ad}")
        print("=" * 60)

        cmd = [
            sys.executable, "main.py",  # 使用 sys.executable 确保调用当前 Python 环境
            "--frontend", f,
            "--mutation", m,
            "--align", al,
            "--adv", ad,
            "--epoch", "50"
        ]

        try:
            # 关键修改：使用 Popen 实时流式读取输出
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

            acc = 0.0
            # 实时将 main.py 的 print 打印到当前终端
            for line in process.stdout:
                print(line, end='')  # 实时打印 Epoch 进度或报错信息
                if "FINAL_ACCURACY:" in line:
                    acc = float(line.split(":")[1])

            process.wait()  # 等待子进程彻底结束

            if process.returncode == 0 and acc > 0:
                print(f"\n👉 组合 [{f}+{m}+{al}+{ad}] 跑分结果: {acc:.2f}%")
                results.append({'Frontend': f, 'Mutation': m, 'Alignment': al, 'Adversarial': ad, 'Accuracy': acc})
            else:
                print(f"\n❌ 该组合运行崩溃或未返回准确率！(Return Code: {process.returncode})")

        except Exception as e:
            print(f"\n❌ 启动进程失败: {e}")

        # 每跑完一个，实时存一次档，防止中间断电白跑
        pd.DataFrame(results).to_csv("NAS_Results_temp.csv", index=False)

    # 最终保存排行榜
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by='Accuracy', ascending=False)
        df.to_csv("NAS_Results_Final.csv", index=False)

        print("\n" + "🔥" * 25)
        print(f"🎉 搜索完成！总耗时: {(time.time() - start_time) / 60:.2f} 分钟")
        print("🏆 最强版本答案是：")
        print(df.iloc[0])
        print("🔥" * 25)
    else:
        print("\n⚠️ 所有组合均运行失败，请检查报错日志。")


if __name__ == "__main__":
    main()