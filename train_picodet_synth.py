# 训练入口脚本
# 用法（在 picodet_train 的上级目录运行）：
#   python picodet_train/train_picodet_synth.py
#
# 常用参数：
#   --config   configs/picodet/picodet_s_640_synth_v2.yml  指定训练配置文件（默认已是最新一轮）
#   --epoch    80       训练轮数
#   --batch-size 8      批大小（显存不够时改成 4）
#   --base-lr  0.0003   初始学习率
#   --clean-output      训练前删除旧的输出目录，重新开始
#   --resume   output/picodet_s_640_synth_v2/epoch_40.pdparams  从断点继续训练
#   --amp                开启混合精度训练（速度更快，显存占用更少）
#
# 训练结果保存在：
#   PaddleDetection/output/picodet_s_640_synth_v2/best_model.pdparams

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


def sync_configs(src: Path, paddle_det: Path):
    # 将 picodet_train/configs/ 下的自定义配置同步到 PaddleDetection/configs/ 对应目录
    for src_file in src.rglob("*.yml"):
        dst = paddle_det / "configs" / src_file.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst)


def run(cmd, cwd, env):
    print(" ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def read_save_dir(config_path: Path) -> str:
    # 从配置文件中读取 save_dir，用于定位训练输出目录
    text = config_path.read_text(encoding="utf-8")
    matches = re.findall(r"(?m)^save_dir:\s*(.+?)\s*$", text)
    if matches:
        return matches[-1].strip().strip("'\"")
    return "output/picodet_s_640_synth_v2"


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train PicoDet-S on the synthetic VOC dataset.")
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-lr", type=float, default=0.0003)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training.")
    parser.add_argument("--resume", default="", help="Checkpoint path for resume training.")
    parser.add_argument("--config", default="configs/picodet/picodet_s_640_synth_v2.yml")
    parser.add_argument("--clean-output", action="store_true", help="Delete old training output before training.")
    parser.add_argument("--keep-checkpoints", action="store_true", help="Keep all checkpoints after training.")
    args = parser.parse_args()

    # 虚拟环境中的 Python，包含 PaddlePaddle 和所有依赖
    python = root / ".venv_gpu" / "Scripts" / "python.exe"
    paddle_det = root / "PaddleDetection"
    dataset = root / "dataset" / "trainval.txt"
    cache = root / ".cache"

    if not python.exists():
        raise FileNotFoundError(python)
    if not paddle_det.exists():
        raise FileNotFoundError(paddle_det)
    if not dataset.exists():
        raise FileNotFoundError(dataset)

    # 把 configs/ 里的配置文件同步到 PaddleDetection 再开始训练
    sync_configs(root / "configs", paddle_det)

    config = paddle_det / args.config
    if not config.exists():
        raise FileNotFoundError(config)

    output_dir = paddle_det / read_save_dir(config)
    if "v2" not in args.config and "finetune" in args.config.replace("\\", "/"):
        base_weights = paddle_det / "output" / "picodet_s_640_synth_finetune_hardneg" / "best_model.pdparams"
        if not base_weights.exists():
            raise FileNotFoundError(f"Fine-tune base weights not found: {base_weights}")

    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)

    cache.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HOME"] = str(root)
    env["USERPROFILE"] = str(root)
    env["XDG_CACHE_HOME"] = str(cache)

    cmd = [
        str(python),
        "tools/train.py",
        "-c",
        args.config,
        "--eval",
        "-o",
        "use_gpu=true",
        f"epoch={args.epoch}",
        f"TrainReader.batch_size={args.batch_size}",
        "EvalReader.batch_size=4",
        f"LearningRate.base_lr={args.base_lr}",
    ]
    if args.resume:
        cmd.extend(["-r", args.resume])
    if args.amp:
        cmd.append("--amp")

    run(cmd, cwd=paddle_det, env=env)

    # 训练结束后只保留 best_model，删除中间 checkpoint 节省磁盘空间
    if not args.keep_checkpoints:
        for path in output_dir.iterdir():
            if not path.name.startswith("best_model"):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        print(f"Kept best_model files in: {output_dir}")


if __name__ == "__main__":
    main()
