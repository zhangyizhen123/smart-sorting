# 公共数据集核查清单

核查日期：2026-09-29。

没有找到一个公开数据集同时满足本项目的 9 种形状、10 种颜色、躺着的圆柱、长方体、40 mm 物块、固定斜视相机和比赛托盘场景。因此公共数据只能用于预训练、代码验证和负样本补充，最终模型必须用项目自己的 MaixCAM 2 实拍数据校准。

## 优先下载

### 1. Cubes, craters and cylinder detection

- 平台：[Kaggle](https://www.kaggle.com/datasets/saikatpanda/cubes-craters-and-cylinder)
- 标识：`saikatpanda/cubes-craters-and-cylinder`
- 许可证：CC0 Public Domain
- 数据情况：约 1.3 GB，640×640 JPG，YOLO 检测标注；包含黄色/绿色立方体、红色圆柱和干扰物。
- 用途：最接近本任务的检测预训练和 YOLO 数据链路验证。
- 局限：不是实例分割；没有长方体、9 类物块、10 种颜色和你们的躺姿/单斜视条件。

```bash
mkdir -p external_data/cubes_cylinder
kaggle datasets download \
  -d saikatpanda/cubes-craters-and-cylinder \
  -p external_data/cubes_cylinder --unzip
```

### 2. cubes&cylinders

- 平台：[Kaggle](https://www.kaggle.com/datasets/mrpoulpe/cubes-and-cylinders)
- 标识：`mrpoulpe/cubes-and-cylinders`
- 许可证：Apache-2.0
- 数据情况：约 27 MB，主题为 cube/cylinder 检测。
- 用途：快速检查类别映射、标签转换、训练启动和推理脚本。
- 局限：数据规模和说明较少，不能作为主训练集；下载后必须检查图片、标签和类别分布。

```bash
mkdir -p external_data/cubes_cylinders
kaggle datasets download \
  -d mrpoulpe/cubes-and-cylinders \
  -p external_data/cubes_cylinders --unzip
```

### 3. Geometric Shapes Dataset

- 平台：[Hugging Face](https://huggingface.co/datasets/0-ma/geometric-shapes)
- 标识：`0-ma/geometric-shapes`
- 许可证：MIT
- 数据情况：14,694 train、2,100 validation、4,200 test；50×50 RGB 合成分类图；类别包括 circle、triangle、square、pentagon、hexagon 和无形状。
- 用途：验证图像分类、增强、TensorBoard/WandB 和数据加载代码。
- 局限：二维、低分辨率、没有圆柱/长方体和实例框，不能直接训练比赛模型。

```bash
python -m pip install -U huggingface_hub datasets
huggingface-cli download 0-ma/geometric-shapes \
  --repo-type dataset --local-dir external_data/geometric_shapes
```

Python 加载方式：

```python
from datasets import load_dataset

ds = load_dataset("0-ma/geometric-shapes")
print(ds)
```

### Images of primitive 3D objects for classification

- 平台：[Kaggle](https://www.kaggle.com/datasets/sirsolim/images-of-primitive-3d-objects-for-classification)
- 标识：`sirsolim/images-of-primitive-3d-objects-for-classification`
- 许可证：CC0 Public Domain
- 数据情况：约 480,000 张 Blender 合成的 40×40 灰度图，四类为 `cube`、`sphere`、`cone`、`torus`；同时提供 one-hot 标签和 CSV/NumPy 文件。
- 适合：检查分类器、数据读取、训练循环和日志监控。
- 不适合：没有 `cylinder`、`cuboid`，不是检测/分割数据，也没有真实相机、颜色、纹理、躺姿或双视角。不能用它证明长方体与躺圆柱可分。

```bash
mkdir -p external_data/primitive_3d
kaggle datasets download \
  -d sirsolim/images-of-primitive-3d-objects-for-classification \
  -p external_data/primitive_3d --unzip
```

## 可选补充

### CLEVR

- 官方页面：[Stanford CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/)
- 下载地址：`https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip`
- 许可证：CC BY 4.0；引用 Johnson et al., 2017。
- 数据情况：约 18 GB 的合成 RGB 场景，包含 cubes、spheres、cylinders；同时提供场景 JSON、物体属性和分割信息。
- 用途：生成圆柱/方块在不同位置、遮挡、颜色和光照下的合成预训练样本；验证从场景标注转换到检测/分割标注的代码。
- 局限：cube 不是比赛中的任意长方体；渲染风格与 3D 打印件有域差异；需要自行从 segmentation/scene JSON 生成 YOLO 标签。

```bash
mkdir -p external_data/clevr
wget -c https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip \
  -O external_data/clevr/CLEVR_v1.0.zip
unzip -q external_data/clevr/CLEVR_v1.0.zip -d external_data/clevr
```

### BOP / T-LESS

- 官方页面：[BOP datasets](https://bop.felk.cvut.cz/datasets/)；[T-LESS](https://cmp.felk.cvut.cz/t-less/)
- 内容：真实工业物体的 RGB/RGB-D 图像、CAD 模型、6D 姿态、遮挡场景和可见/完整掩膜。T-LESS 包含 30 个无纹理工业物体，并有多个同步传感器。
- 许可证：按数据集分别确认；T-LESS 页面标明 CC BY 4.0。BOP 中其他数据集可能有不同条款。
- 用途：补充真实材质、遮挡、背景和深度信息；可用于评估实例分割与遮挡处理方法。
- 局限：标签是物体 ID 和姿态，不是 `cylinder/cuboid` 语义类；需要将 BOP 掩膜/框转换为 YOLO 格式，并确认物体几何是否真的对应目标类别。

```bash
python -m pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download bop-benchmark/tless \
  --repo-type dataset --local-dir external_data/tless
```

也可以按官方页面下载 `tless_base.zip`、`tless_models.zip`、`tless_test_all.zip` 和 `tless_train_pbr.zip`。商业发布前逐一保留对应 README 和许可证。

### Kaggle Geometric Shapes Mathematics

- 平台：[Kaggle](https://www.kaggle.com/datasets/reevald/geometric-shapes-mathematics)
- 标识：`reevald/geometric-shapes-mathematics`
- 许可证：CC0 Public Domain
- 规模：8 类二维形状，每类约 1,500 train、500 validation、500 test，224×224 RGB。
- 用途：二维轮廓分类的预训练和增强测试。
- 不能识别三维物块，也不能解决躺圆柱/长方体。

```bash
mkdir -p external_data/geometric_shapes_math
kaggle datasets download \
  -d reevald/geometric-shapes-mathematics \
  -p external_data/geometric_shapes_math --unzip
```

### ShapeNetCore

- 平台：[Hugging Face](https://huggingface.co/datasets/ShapeNet/ShapeNetCore)
- 内容：约 51,300 个三维模型、55 个常见类别；可以筛选 cylinder、cube 等模型后用 Blender/BlenderProc 按固定斜视角渲染合成数据。
- 访问：需要手动接受 ShapeNet 条款，仓库为 gated；许可证显示为 `other`，不能直接视为商业可用。
- 用途：在自采数据不足时做姿态、光照、背景和遮挡的合成预训练。
- 风险：合成到真实 3D 打印物块存在域差异，必须用真实数据微调；商业发布前需要逐项确认 ShapeNet 条款。

```bash
python -m pip install -U huggingface_hub
huggingface-cli login
huggingface-cli download ShapeNet/ShapeNetCore \
  --repo-type dataset --local-dir external_data/shapenetcore
```

## 不建议当主训练数据

- COCO：许可证和下载公开，但类别中没有 `cuboid`、`cylinder` 这类与你们任务对应的几何标签，适合通用 backbone 预训练，不能直接解决本任务。
- 仅有表格几何特征的数据集：能训练传统分类器，却没有图像、姿态、材质和相机成像信息。
- 许可证为 `Unknown` 或没有数据卡的数据集：不纳入正式训练集，避免无法确认商用权利。

## 下载后统一检查

目录名称可能因数据集压缩包而不同，先用下面的命令确认实际结构和类别文件：

```bash
find external_data -maxdepth 4 -type f | head -80
find external_data -iname 'data.yaml' -o -iname '*.names' -o -iname 'classes.txt'
```

只有确认了原数据集的类别顺序后，才使用 `prepare_yolo.py`，并为该数据集单独建立 class file。不要直接把公开集和自采集随机混合切分；同一物体、同一拍摄序列和同一渲染场景必须放在同一个 group 中，避免验证集泄漏。公开数据的类别定义也不能直接套用项目的 9 类顺序。

## 最终使用建议

1. 用 `saikatpanda/cubes-craters-and-cylinder` 验证 YOLO 数据与检测流程。
2. 用 `0-ma/geometric-shapes` 验证分类训练、日志和数据增强。
3. 只把公开数据作为 warm-start 或负样本；部署模型必须用项目自采的固定斜视分割数据训练和验证，并确认 MaixCAM 2 SDK 支持模型格式与算子。
4. 对长方体/躺圆柱建立独立 hard-case 测试集，按 `cylinder_to_cuboid` 和 `cuboid_to_cylinder` 分别验收。
