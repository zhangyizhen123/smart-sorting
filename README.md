# 智能分拣 · 视觉识别子系统

面向中国大学生工程实践与创新能力大赛「智能分拣」赛项的视觉识别与分拣控制部分。目标是识别 **9 种形状 × 10 种颜色**，并把每轮出现的 6 种 `(shape, color)` 组合动态映射到 6 个储物盒；视觉部署目标为 Sipeed MaixCAM 2，运动控制目标为 STM32。

当前阶段：**九类 STL 合成实例分割基线已完成；托盘/实物数据采集与真机验证尚未开始。**

## 文档导航

| 文档 | 内容 |
|---|---|
| [Excel 任务跟踪表（推荐阅读）](docs/任务跟踪表.xlsx) / [文字版](docs/任务跟踪表.md) | Excel 阅读请打开此工作簿：按进度、资料分工、验收、证据和已完成事项分表，并设有列宽、自动换行、筛选和冻结窗格。`docs/任务跟踪表.csv` 是含完整字段的维护数据源，直接用 Excel 打开时不会保留这些版式。 |
| [队员 GitHub 使用说明](docs/队员GitHub使用说明.md) | 面向不熟悉 GitHub 的队员：查看进度、下载资料、上传文件和反馈任务进展。 |
| [新 Agent 交接文档](docs/Agent交接.md) | 项目约定、当前状态、接手顺序、可独立推进项、所需硬件输入和验证命令 |
| [项目总结](docs/项目总结.md) | 项目全景、实验数据、瓶颈与下一步（建议先读） |
| [视觉训练计划](docs/视觉训练计划.md) | 端到端训练与部署方案（主方案） |
| [九类 STL 首版训练结果](docs/视觉训练结果_STL首版.md) | 合成测试指标、类别短板、预测拼图和使用边界 |
| [老师演示操作说明](docs/老师演示操作说明.md) / [中文演示页面](docs/stl_shape_v1_demo/index.html) | 离线切换 12 张样例、对照参考/预测轮廓；双击根目录“演示模型.cmd”可重新推理 |
| [STL 合成训练步骤](vision/STL合成训练.md) | 数据生成、训练和测试集评估的复现命令 |
| [视觉识别物块清单](docs/视觉识别物块清单.md) | 类别表、俯视易混点、颜色识别要点 |
| [MaixCAM 2 与 STM32 实施方案](docs/MaixCAM2与STM32实施方案.md) | 固定斜视单摄像头、UART、动态分盒和光电投放闭环 |
| [vision/DATASETS.md](vision/DATASETS.md) | 公共数据集核查与许可证边界 |
| [vision/stage1/AUTOML_PLAN.md](vision/stage1/AUTOML_PLAN.md) | 基线归因与调优记录 |
| [models/blocks_40mm/README.md](models/blocks_40mm/README.md) | 40 mm 训练物块 STL 说明 |

## 目录结构

```text
vision/              训练与数据脚本、阶段一代码、配置
models/blocks_40mm/  40 mm 物块 STL、生成脚本、几何校验报告
models/trained/      已归档的九类 STL 合成基线权重
runs/stage1/         实验指标 JSON（权重与日志不入库）
```

`reference/`（约 17 GB 第三方资料）与 `dataset/`（可由脚本复现的合成图）不在版本管理范围内，原因见 `docs/项目总结.md` 第 7 节。

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow numpy tensorboard

python vision/stage1/make_synthetic.py --out dataset/stage1_synthetic --train 240 --val 60 --test 60 --seed 42
python vision/stage1/train_classifier.py --data dataset/stage1_synthetic --epochs 12 --imgsz 128 --device cpu --name stage1_synthetic
```

## 状态提示

阶段一报告的高准确率均来自**合成数据**，只用于验证训练链路，不代表真实物块精度。当前运行时已按固定斜视单摄像头和 MaixCAM 2 → UART → STM32 闭环实现接口骨架；正式指标须以最终相机实拍数据和独立拍摄日测试集为准。
