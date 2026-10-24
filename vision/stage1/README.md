# 阶段一：单图四分类

类别固定为 `cuboid`（长方体）、`tetrahedron`（正四面体）、`prism`（柱体，暂不细分）和 `sphere`（球）。当前目录中的合成数据只用于验证训练链路；正式模型必须用最终相机拍摄的真实物块替换。

数据目录采用 ImageFolder 格式：

```text
dataset/stage1/
  train/cuboid/*.jpg
  train/tetrahedron/*.jpg
  train/prism/*.jpg
  train/sphere/*.jpg
  val/...
  test/...
```

生成可复现的合成 smoke 数据：

```bash
python vision/stage1/make_synthetic.py \
  --out dataset/stage1_synthetic \
  --train 240 --val 60 --test 60 --seed 42
```

安装 CPU 环境并训练：

```bash
python3 -m venv .venv-stage1
source .venv-stage1/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install pillow numpy tensorboard

python vision/stage1/train_classifier.py \
  --data dataset/stage1_synthetic \
  --epochs 12 --batch 32 --imgsz 128 \
  --device cpu --name stage1_synthetic
```

查看日志：

```bash
tensorboard --logdir runs/stage1
```

真实数据采集要求：同一物块的连续帧放在同一个 `group_id`，训练/验证/测试按物块和拍摄批次分组，不能随机拆相邻帧。每类先采集至少 300 张，测试集至少包含一个独立拍摄日。

阶段二可复用本阶段的 `encoder` 权重，把两个相机图像分别送入同一个编码器，再拼接两路特征后增加融合分类头。

本机 smoke 记录（2026-09-29）：使用 80/20/20 张/类的合成数据、CPU、8 epochs，最佳验证准确率 98.75%，测试准确率 92.50%。测试混淆主要发生在 `sphere` 与 `prism`；该结果只用于确认代码、权重保存和 TensorBoard 日志正常。
