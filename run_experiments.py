import os
import sys
import subprocess

# ================= 配置区域 =================
# Python解释器路径 (如果你用了虚拟环境，请确认这里调用的是对的python)
PYTHON_EXEC = sys.executable

# 数据集根目录 (根据你的目录结构 D:\LAST_WORKS\CODE\data\PU\PU_processed)
DATA_ROOT = os.path.join("data", "PU", "PU_processed")

# 定义 PU 数据集的三个工况文件夹名
DOMAINS = ["N15_M07", "N09_M07", "N15_M01"]

# 基础参数
BATCH_SIZE = 40
EPOCHS = 300
LR = 0.0005
CLASSES = 3


# ===========================================

def run_cmd(cmd):
    print(f"\n[Running Command]: {cmd}")
    # 使用 subprocess 调用，确保实时输出显示在控制台
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    if process.returncode != 0:
        print(f"[Error] Experiment failed!")
        sys.exit(1)


def main():
    print(">>> 开始 PU 数据集全工况实验 (Leave-One-Domain-Out)...")

    for source in DOMAINS:
        # 确定 Source 和 Targets
        targets = [d for d in DOMAINS if d != source]

        # 构建路径
        source_path = os.path.join(DATA_ROOT, source)
        target_paths = [os.path.join(DATA_ROOT, t) for t in targets]

        # 检查路径是否存在
        if not os.path.exists(source_path):
            print(f"[Error] Source path not found: {source_path}")
            continue

        target_str = " ".join(target_paths)

        print(f"\n{'=' * 60}")
        print(f"Experiment: Source Domain = {source}")
        print(f"Target Domains = {targets}")

        # 拼接命令行
        # 注意：这里调用 main.py，确保 main.py 在同一目录下
        cmd = (
            f'"{PYTHON_EXEC}" main.py '
            f'--mode bdc '
            f'--source_path "{source_path}" '
            f'--target_paths {target_str} '
            f'--batch_size {BATCH_SIZE} '
            f'--classes {CLASSES}'
        )

        run_cmd(cmd)

    print("\n>>> 所有实验运行完毕！")


if __name__ == "__main__":
    main()