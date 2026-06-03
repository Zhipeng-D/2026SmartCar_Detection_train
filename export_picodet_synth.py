# 导出入口脚本——将训练好的权重转换为推理模型
# 用法（在 picodet_train 的上级目录运行）：
#   python picodet_train/export_picodet_synth.py
#
# 常用参数：
#   --weights     output/picodet_s_640_synth_v2/best_model.pdparams  要导出的权重文件
#   --config      configs/picodet/picodet_s_640_synth_v2.yml         对应的配置文件
#   --output-dir  output_inference/picodet_s_640_synth_v2            导出结果保存目录
#
# 导出结果包含三个文件：
#   model.pdmodel        模型结构
#   model.pdiparams      模型参数
#   infer_cfg.yml        推理配置（含类别列表、预处理参数），部署时需要
#
# 导出后可用于 RK3588 等嵌入式平台的推理部署

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
    parser = argparse.ArgumentParser(description="Export trained PicoDet-S weights for inference.")
    parser.add_argument("--weights", default="output/picodet_s_640_synth_v2/best_model.pdparams")
    parser.add_argument("--config", default="configs/picodet/picodet_s_640_synth_v2.yml")
    parser.add_argument("--output-dir", default="output_inference/picodet_s_640_synth_v2")
    args = parser.parse_args()

    # 虚拟环境中的 Python，包含 PaddlePaddle 和所有依赖
    python = root / ".venv_gpu" / "Scripts" / "python.exe"
    paddle_det = root / "PaddleDetection"
    cache = root / ".cache"

    # 把 configs/ 里的配置文件同步到 PaddleDetection 再开始导出
    sync_configs(root / "configs", paddle_det)

    env = os.environ.copy()
    env["HOME"] = str(root)
    env["USERPROFILE"] = str(root)
    env["XDG_CACHE_HOME"] = str(cache)

    cmd = [
        str(python),
        "tools/export_model.py",
        "-c",
        args.config,
        "-o",
        f"weights={args.weights}",
        "--output_dir",
        args.output_dir,
    ]
    print(" ".join(cmd))
    raise SystemExit(subprocess.run(cmd, cwd=str(paddle_det), env=env).returncode)


if __name__ == "__main__":
    main()
