# 智能分拣 · 视觉识别子系统

面向中国大学生工程实践与创新能力大赛「智能分拣」赛项的视觉识别部分。目标是识别 **9 种形状 × 10 种颜色**，并把 `(shape, color)` 映射到储物盒；部署目标为树莓派。

当前阶段：**阶段一（合成数据链路验证）已完成，真实数据采集尚未开始。**

## 文档导航

| 文档 | 内容 |
|---|---|
| [项目总结.md](项目总结.md) | 项目全景、实验数据、瓶颈与下一步（建议先读） |
| [视觉训练计划.md](视觉训练计划.md) | 端到端训练与部署方案（主方案） |
| [视觉识别物块清单.md](视觉识别物块清单.md) | 类别表、俯视易混点、颜色识别要点 |
| [vision/DATASETS.md](vision/DATASETS.md) | 公共数据集核查与许可证边界 |
| [vision/stage1/AUTOML_PLAN.md](vision/stage1/AUTOML_PLAN.md) | 基线归因与调优记录 |
| [models/blocks_40mm/README.md](models/blocks_40mm/README.md) | 40 mm 训练物块 STL 说明 |

## 目录结构

```text
vision/              训练与数据脚本、阶段一代码、配置
models/blocks_40mm/  40 mm 物块 STL、生成脚本、几何校验报告
runs/stage1/         实验指标 JSON（权重与日志不入库）
```

`reference/`（约 17 GB 第三方资料）与 `dataset/`（可由脚本复现的合成图）不在版本管理范围内，原因见 `项目总结.md` 第 7 节。

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow numpy tensorboard

python vision/stage1/make_synthetic.py --out dataset/stage1_synthetic --train 240 --val 60 --test 60 --seed 42
python vision/stage1/train_classifier.py --data dataset/stage1_synthetic --epochs 12 --imgsz 128 --device cpu --name stage1_synthetic
```

## 状态提示

阶段一报告的高准确率均来自**合成数据**，只用于验证训练链路，不代表真实物块精度。正式指标须以双相机实拍数据与独立拍摄日测试集为准。
