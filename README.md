# 智能分拣 · 视觉识别子系统

面向中国大学生工程实践与创新能力大赛「智能分拣」赛项的视觉识别与分拣控制部分。目标是识别 **9 种形状 × 10 种颜色**，并把每轮出现的 6 种 `(shape, color)` 组合动态映射到 6 个储物盒；视觉部署目标为 Sipeed MaixCAM 2，运动控制目标为 STM32。

当前阶段：**联合优化流程已在同一批 5800 张新合成场景上同时改善识别、误报与轮廓，默认演示已更新：严格实例得分 86.01%，额外误报 75，总体边界 IoU 0.8541。46 张已有实拍属于训练/开发材料，独立实拍与整机分拣验收尚未完成，未达到真实场景 99% 的证明标准。**

## 文档导航

| 文档 | 内容 |
|---|---|
| [Excel 任务跟踪表（推荐阅读）](docs/任务跟踪表.xlsx) / [文字版](docs/任务跟踪表.md) | Excel 阅读请打开此工作簿：按进度、资料分工、验收、证据和已完成事项分表，并设有列宽、自动换行、筛选和冻结窗格。`docs/任务跟踪表.csv` 是含完整字段的维护数据源，直接用 Excel 打开时不会保留这些版式。 |
| [队员 GitHub 使用说明](docs/队员GitHub使用说明.md) | 面向不熟悉 GitHub 的队员：查看进度、下载资料、上传文件和反馈任务进展。 |
| [Claude / Opus 接手入口](CLAUDE.md) / [新 Agent 交接文档](docs/Agent交接.md) | 另一台 Windows 的克隆安装、锁定 CPU 环境、首次检查、接手提示词与当前项目状态 |
| [项目总结](docs/项目总结.md) | 项目全景、实验数据、瓶颈与下一步（建议先读） |
| [视觉训练计划](docs/视觉训练计划.md) | 端到端训练与部署方案（主方案） |
| [联合优化模型验收结果](docs/联合优化模型验收结果.md) | 当前默认流程；5800 张新合成场景三版共同对照、逐类短板、正式演示与真实数据限制 |
| [九类 STL 首版训练结果](docs/视觉训练结果_STL首版.md) | 合成测试指标、类别短板、预测拼图和使用边界 |
| [三组易混形状专项训练](docs/三组易混形状专项训练.md) | 困难场景、冻结数据、严格实例评分、复核器和 CPU/正式训练配置；99%是否达成以实验结果为准 |
| [三组专项实际结果](docs/视觉训练结果_三组专项.md) | 新冻结测试严格得分 57.10%→93.62%；三组双向错分及旧测试集 72.16%→65.69% 的退化；专项版作为试验入口 |
| [老师演示操作说明](docs/老师演示操作说明.md) / [中文演示页面](docs/stl_shape_v1_demo/index.html) | 双击根目录“演示模型.cmd”，可以自行导入图片并识别形状；HTML 文件可离线查看 12 张预置样例 |
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
models/trained/      联合优化默认流程、首版权重与三组专项对照包
runs/stage1/         实验指标 JSON（权重与日志不入库）
```

`reference/`（约 17 GB 第三方资料）与 `dataset/`（可由脚本复现的合成图）不在版本管理范围内，原因见 `docs/项目总结.md` 第 7 节。

## 快速开始

**另一台 Windows 首次接手**：按[交接文档](docs/Agent交接.md)使用 Python 3.14 新建 `.venv-vision`，安装 `vision/requirements-handoff.txt`，再运行测试和演示。代码、三版模型及样例已入库；训练数据与原机器的虚拟环境需另行准备。

查看当前版本：双击根目录 **演示模型.cmd**，等页面显示“本机模型已就绪”，选择图片后点击“识别形状”。保留启动窗口，最低置信度先保持 0.25；可切换查看校正轮廓和模型原始轮廓，无需先手工标注。**演示首版对照.cmd** 和 **演示专项模型.cmd** 保留两版旧流程。网页预置样例仍为首版；上传图片使用新流程，各版均未完成独立实拍验收。

下面保留早期分类基线的复现命令；专项分割训练请使用[专项说明](docs/三组易混形状专项训练.md)中的命令。

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow numpy tensorboard

python vision/stage1/make_synthetic.py --out dataset/stage1_synthetic --train 240 --val 60 --test 60 --seed 42
python vision/stage1/train_classifier.py --data dataset/stage1_synthetic --epochs 12 --imgsz 128 --device cpu --name stage1_synthetic
```

## 状态提示

阶段一报告的高准确率均来自**合成数据**，只用于验证训练链路，不代表真实物块精度。当前运行时已按固定斜视单摄像头和 MaixCAM 2 → UART → STM32 闭环实现接口骨架；正式指标须以最终相机实拍数据和独立拍摄日测试集为准。
