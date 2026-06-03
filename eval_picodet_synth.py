# 评估入口脚本——在测试集上计算 mAP 等精度指标
# 用法（在 picodet_train 的上级目录运行）：
#   python picodet_train/eval_picodet_synth.py
#
# 常用参数：
#   --weights  output/picodet_s_640_synth_v2/best_model.pdparams  指定要评估的权重
#   --config   configs/picodet/picodet_s_640_synth_v2.yml         对应的配置文件
#
# 输出结果：各类别 AP 和整体 mAP，打印在终端

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def sync_configs(src: Path, paddle_det: Path):
    # 将 picodet_train/configs/ 下的自定义配置同步到 PaddleDetection/configs/ 对应目录
    for src_file in src.rglob("*.yml"):
        dst = paddle_det / "configs" / src_file.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate PicoDet-S synthetic dataset weights.")
    parser.add_argument("--weights", default="output/picodet_s_640_synth_v2/best_model.pdparams")
    parser.add_argument("--config", default="configs/picodet/picodet_s_640_synth_v2.yml")
    args = parser.parse_args()

    # 虚拟环境中的 Python，包含 PaddlePaddle 和所有依赖
    python = root / ".venv_gpu" / "Scripts" / "python.exe"
    paddle_det = root / "PaddleDetection"
    cache = root / ".cache"

    # 把 configs/ 里的配置文件同步到 PaddleDetection 再开始评估
    sync_configs(root / "configs", paddle_det)

    env = os.environ.copy()
    env["HOME"] = str(root)
    env["USERPROFILE"] = str(root)
    env["XDG_CACHE_HOME"] = str(cache)

    cmd = [
        str(python),
        "tools/eval.py",
        "-c",
        args.config,
        "-o",
        f"weights={args.weights}",
        "use_gpu=true",
    ]
    print(" ".join(cmd))
    raise SystemExit(subprocess.run(cmd, cwd=str(paddle_det), env=env).returncode)


if __name__ == "__main__":
    main()
