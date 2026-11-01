# 基于 STL 的第一版形状训练

当前没有托盘和固定相机时，可以先用 `models/blocks_40mm/` 下的九个 STL 生成合成实例分割数据，训练九类形状模型。它用来验证建模类别、自动标签和训练流程，并提供之后实拍微调的初始化权重；不能据此宣称实物准确率、颜色准确率、托盘坐标精度或 MaixCAM2 部署成功。

## 自动与人工工作

| 工作 | 自动化 | 说明 |
|---|---|---|
| STL 读取、随机摆姿、渲染、形状多边形标签 | 自动 | 多物块放在分离区域；标签由投影轮廓生成，不需要手动画分割边界 |
| 数据切分、YOLO 格式检查、训练、指标和预测图 | 自动 | 训练集/验证集/测试集按场景分组 |
| 模型类别是否与实物一致 | 需要人工核对 | STL 只表达几何，不证明打印物或比赛物块的形状定义 |
| 合成图是否像最终设备画面 | 需要人工检查 | 现在没有托盘、镜头、灯光和实际视角，渲染只能覆盖通用斜视变化 |
| 颜色、抓取坐标、遮挡和接触物块的标签 | 这版不能完成 | STL 不含材料颜色；没有实物托盘无法标定坐标。正式实拍阶段要采样颜色并人工抽查歧义掩膜 |

## 一次性安装

在项目根目录打开 PowerShell。当前电脑没有 NVIDIA 独立显卡，因此先安装 CPU 版 PyTorch。若安装器提示当前 Python 版本没有兼容的 PyTorch wheel，需要先安装 Python 3.12 x64，再用 `py -3.12` 重建环境。

```powershell
cd E:\智能分拣
python -m venv .venv-vision
.\.venv-vision\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install "ultralytics>=8.3.0"
```

## 生成数据并训练

第一轮生成每类 120 个目标场景，共 1080 张图；每张含 1～4 个不互相遮挡的物块。按场景分为 864 张训练、108 张验证、108 张测试图。STL 是 40 mm 建模件，渲染器随机改变姿态、图像大小、背景、物体颜色和斜视角，并自动导出 YOLO 分割多边形。

```powershell
python vision\scripts\make_stl_synthetic.py `
  --out dataset\stl_synthetic_raw `
  --per-class 120 `
  --image-size 320 `
  --seed 42

python vision\scripts\prepare_yolo.py `
  --images dataset\stl_synthetic_raw\images `
  --labels dataset\stl_synthetic_raw\labels `
  --out dataset\stl_synthetic_yolo `
  --names vision\configs\classes.txt `
  --seed 42

python vision\scripts\train_seg.py `
  --data dataset\stl_synthetic_yolo\data.yaml `
  --model yolo11n-seg.pt `
  --name stl_shape_v1 `
  --project runs\segment `
  --epochs 20 `
  --imgsz 320 `
  --batch 4 `
  --workers 0 `
  --cache none `
  --device cpu `
  --patience 8 `
  --no-amp
```

`yolo11n-seg.pt` 首次使用需要从 Ultralytics 官方发布源下载。脚本现在把相对数据和输出目录固定解析到项目根目录，因此从其他工作目录启动也不会把 `runs/segment` 再套一层。若 CPU 训练太慢，可把环境迁移到带 NVIDIA GPU 的机器后使用 `--device 0 --batch 8`；训练输入、数据和随机种子保持一致。

本轮实际配置为 YOLO11n-seg、20 轮、320 输入、batch 4、CPU、seed 42。已归档的最佳权重位于 `models/trained/stl_shape_v1/best.pt`。

## 检查结果

- 数据集：`dataset/stl_synthetic_yolo/manifest.csv`、`data.yaml`。
- 本轮训练曲线与配置：`runs/segment/runs/segment/stl_shape_v1/`；最佳权重已复制到上方归档路径。
- 固定测试集评估：运行 `python vision\scripts\evaluate_stl_test.py`，指标写入 `docs/视觉训练指标_STL首版.json`，完整预测图写入 `runs/segment/stl_shape_v1_test_verified_predictions/`。
- 老师展示用说明和拼图：`docs/视觉训练结果_STL首版.md`、`docs/视觉训练首版预测拼图.jpg`。
- 中文演示：`docs/stl_shape_v1_demo/index.html` 可离线查看样例；双击项目根目录 `演示模型.cmd` 启动本机模型后可导入自己的图片并识别。详细步骤见 `docs/老师演示操作说明.md`。
- 测试集 mask mAP50 为 0.915、mAP50-95 为 0.872。主要混淆是 `cube/cuboid`；请结合逐类指标和预测图解释，不能只展示总分。

合成测试集只衡量模型对这套渲染规则的拟合程度。拿到托盘、实物和最终固定相机后，要重新采集真实图像、人工抽查自动生成的掩膜、冻结独立测试集，并用实拍数据微调或重训；之后才能评估真实识别和抓取表现。
