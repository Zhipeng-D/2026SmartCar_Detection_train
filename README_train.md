# PicoDet-S 智能车目标检测训练工作区

基于 PaddleDetection 的 PicoDet-S 模型，用于检测赛道上的 13 类目标，最终部署在 RK3588 嵌入式平台上。

## 检测类别（13 类）

`coin`（金币）、`model`（小模型）、`car`（小车）、`rgb`（红绿灯）、`traffic1`（交通标志）、
`RoadSign0`、`RoadSign1`（路牌）、`Stone`（石头）、`beginDoor`（起始门）、`beginSign`（起始标志）、
`EndSign`（终止标志）、`crosswalk`（斑马线）、`ward`（区域标记）

---

## 目录结构

```
picodet_train/
├── configs/                        # 自定义配置文件（上传 GitHub 的核心内容）
│   ├── datasets/
│   │   └── synth_voc.yml           # 数据集配置
│   └── picodet/
│       ├── picodet_s_640_synth.yml              # 第1轮：从 COCO 预训练从头训练
│       ├── picodet_s_640_synth_finetune.yml     # 第2轮：fine-tune
│       ├── picodet_s_640_synth_finetune_hardneg.yml  # 第3轮：加入 hard negative
│       └── picodet_s_640_synth_v2.yml           # 第4轮：新数据集（当前使用）
├── dataset/
│   ├── JPEGImages/                 # 训练图片（不上传 GitHub，单独分发）
│   ├── Annotations/                # VOC XML 标注（同上）
│   ├── trainval.txt                # 训练集文件列表
│   ├── test.txt                    # 测试集文件列表
│   └── label_list.txt              # 类别列表
├── PaddleDetection/                # PaddleDetection 框架（不上传 GitHub，自行下载）
├── train_picodet_synth.py          # 训练入口
├── eval_picodet_synth.py           # 评估入口
└── export_picodet_synth.py         # 导出入口
```

---

## 环境准备

### 1. 下载 PaddleDetection

将 PaddleDetection（release/2.6 分支）克隆到 `picodet_train/PaddleDetection/`：

```bash
git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleDetection.git picodet_train/PaddleDetection
```

### 2. 安装依赖

在 `picodet_train/` 下创建虚拟环境并安装依赖：

```bash
cd picodet_train
python -m venv .venv_gpu
.venv_gpu\Scripts\activate
pip install paddlepaddle-gpu
pip install -r PaddleDetection/requirements.txt
```

### 3. 放入数据集

将训练数据解压到 `picodet_train/dataset/`，确保目录结构如下：

```
dataset/
├── JPEGImages/       （训练图片）
├── Annotations/      （VOC XML 标注）
├── trainval.txt
├── test.txt
└── label_list.txt
```

### 4. 验证 GPU

```bash
picodet_train\.venv_gpu\Scripts\python.exe -c "import paddle; print(paddle.__version__); print(paddle.device.get_device()); paddle.utils.run_check()"
```

---

## 训练流程

所有命令在 `picodet_train/` 的**上级目录**运行。

### 第一步：训练

```bash
picodet_train\.venv_gpu\Scripts\python.exe picodet_train\train_picodet_synth.py
```

默认使用 `configs/picodet/picodet_s_640_synth_v2.yml`，从第 3 轮的最优权重开始 fine-tune，
训练 80 个 epoch，结果保存在：

```
PaddleDetection/output/picodet_s_640_synth_v2/best_model.pdparams
```

常用参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epoch` | 训练轮数 | 80 |
| `--batch-size` | 批大小，显存不够时改 4 | 8 |
| `--base-lr` | 初始学习率 | 0.0003 |
| `--clean-output` | 训练前清空旧输出 | 否 |
| `--resume` | 从断点继续训练，填 checkpoint 路径 | 无 |
| `--amp` | 开启混合精度，速度更快显存更少 | 否 |

> RTX 4060 8GB 建议 `--batch-size 8`，若报显存不足改为 `--batch-size 4`。

### 第二步：评估

训练结束后，在测试集上计算各类别 AP 和整体 mAP：

```bash
picodet_train\.venv_gpu\Scripts\python.exe picodet_train\eval_picodet_synth.py
```

### 第三步：导出

将权重导出为推理模型（用于 RK3588 部署）：

```bash
picodet_train\.venv_gpu\Scripts\python.exe picodet_train\export_picodet_synth.py
```

导出结果保存在 `PaddleDetection/output_inference/picodet_s_640_synth_v2/`，包含：

- `model.pdmodel` — 模型结构
- `model.pdiparams` — 模型参数
- `infer_cfg.yml` — 推理配置（类别列表、预处理参数），部署时需要

---

## 从头训练（可选）

如果不想在旧权重基础上 fine-tune，可以从第 1 轮从头开始：

```bash
picodet_train\.venv_gpu\Scripts\python.exe picodet_train\train_picodet_synth.py ^
    --config configs/picodet/picodet_s_640_synth.yml ^
    --epoch 300 --base-lr 0.005 --clean-output
```
