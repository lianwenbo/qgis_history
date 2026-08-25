# 历史地图线要素提取工具

这是一个面向历史地图扫描图的半自动数字化项目，结合传统计算机视觉、深度学习和几何后处理，提取地图中的经纬线与行政边界，并生成可供 LabelMe 和 QGIS 继续处理的数据。

项目当前以 Python 脚本为主，适用于算法实验、标注辅助、模型训练和批量推理，不是完整的桌面 GIS 应用。

## 主要功能

### 经纬线检测

- 使用 Sobel、形态学处理和 Hough 变换检测经线；
- 从图像轮廓中提取和简化纬线弧线；
- 支持 YOLO 实例分割和 UNet 语义分割；
- 支持传统算法与深度学习模型之间的回退切换；
- 根据所选管线输出像素坐标、分割掩膜、折线标注或叠加预览图。

### 标注与训练数据处理

- 解析 LabelMe 的 `line`、`linestrip` 和 `polygon` 标注；
- 将经纬线标注转换为 YOLO 分割格式；
- 将 LabelMe 标注转换为 UNet 语义分割掩膜；
- 对大幅地图进行带重叠的分块处理；
- 支持模型训练和结果评估。

### 几何后处理

- 从分割掩膜提取中心线和骨架；
- 弥合短距离断线；
- 合并方向和位置相近的线段；
- 对折线进行平滑、抽稀和短线过滤；
- 将推理结果转换回 LabelMe 折线标注，便于人工复核。

### QGIS GCP 生成

- 从 LabelMe 经纬线标注中识别经线和纬线；
- 排除插图区域并对重复线进行过滤；
- 根据人工确认的经纬度锚点推断其他网格线度数；
- 计算经纬线交点；
- 输出 QGIS Georeferencer 可读取的 `.points` 控制点文件和预览图。

### 行政边界辅助提取

- 使用 UNet 对行政边界进行训练和滑窗推理；
- 支持一级、二级边界分类；
- 将边界掩膜转换为经过骨架化和简化的 LabelMe 线标注；
- 可利用 CHGIS 矢量边界与 QGIS GCP 生成辅助标注。

## 处理流程

```text
扫描地图
  -> LabelMe 标注或传统算法初检
  -> YOLO / UNet 数据准备
  -> 模型训练与滑窗推理
  -> 骨架化、断线弥合、合并与简化
  -> LabelMe 复核
  -> 经纬线交点计算
  -> QGIS GCP 文件
```

## 环境

建议使用 Python 3.12。安装基础依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install scipy
```

深度学习训练需要可用的 PyTorch 环境。CPU 和 Apple Silicon 用户可以安装：

```bash
python -m pip install torch torchvision
```

CUDA 用户应根据本机 CUDA 版本使用 [PyTorch 官方安装命令](https://pytorch.org/get-started/locally/)。Apple Silicon、CUDA 和 CPU 的设备选择由不同训练脚本分别处理。

CHGIS 辅助标注还需要以下可选依赖：

```bash
python -m pip install geopandas pandas shapely pyogrio
```

## 快速开始

### 检测经纬线

不指定模型时使用传统算法：

```bash
python hybrid_pipeline.py \
  --image /path/to/map.jpg \
  --output output
```

指定 YOLO 分割模型：

```bash
python hybrid_pipeline.py \
  --image /path/to/map.jpg \
  --model /path/to/model.pt \
  --output output
```

该入口在输出目录中生成 `result_<原文件名>` 预览图和 `result_<原文件名>.txt` 像素坐标文件。

### 生成 QGIS 控制点

先提取地图边缘的刻度区域，辅助确认经纬度锚点：

```bash
python generate_gcp.py \
  --image /path/to/map.jpg \
  --json /path/to/annotation.json \
  --extract-scales
```

刻度区域图片当前固定写入系统临时目录，相关实现面向 Unix-like 环境。

确认锚点后生成 `.points` 文件：

```bash
python generate_gcp.py \
  --image /path/to/map.jpg \
  --json /path/to/annotation.json \
  --anchors "v:0=100,1=104;h:0=40,1=36" \
  --output /path/to/map.points
```

锚点格式中的 `v` 表示经线，`h` 表示纬线，数字索引对应脚本输出的排序结果。

### 训练分类行政边界模型

```bash
python boundaries/autodl_train_boundary.py \
  --labelme_dir /path/to/boundary-labelme \
  --model_path /path/to/unet_boundary.pth \
  --epochs 30 \
  --batch_size 16
```

该入口训练背景、一级边界和二级边界三类 UNet 模型。将模型推理结果转换为 LabelMe 折线：

```bash
python boundaries/infer_boundary.py \
  --model /path/to/unet_boundary.pth \
  --image_dir /path/to/images \
  --out_dir /path/to/labelme-output \
  --vis_dir /path/to/preview-output \
  --targets map_name
```

## 核心文件

| 文件 | 作用 |
| --- | --- |
| `demo.py` | 传统经纬线检测示例 |
| `hybrid_pipeline.py` | 传统算法与 YOLO 模型统一推理入口 |
| `data_preparation.py` | LabelMe 到 YOLO 分割数据转换 |
| `unet_data_prep.py` | LabelMe 到 UNet 掩膜及分块数据转换 |
| `train_model.py` | YOLO 分割模型训练 |
| `unet_model.py` | UNet 模型与训练相关实现 |
| `post_processing.py` | 骨架、线段拟合、合并和简化 |
| `generate_gcp.py` | QGIS `.points` 控制点生成 |
| `boundaries/train_boundary.py` | 二分类行政边界 UNet 训练与基础推理 |
| `boundaries/autodl_train_boundary.py` | 一级、二级行政边界三分类 UNet 训练 |
| `boundaries/infer_boundary.py` | 分类边界滑窗推理和 LabelMe 输出 |
| `boundaries/generate_chgis_labelme.py` | CHGIS 边界到扫描图标注的逆向映射 |

`boundaries/generate_chgis_labelme.py` 的默认输入路径仅用于开发环境。公开用户运行时必须显式传入 `--prefecture-data`、`--province-data`、`--image`、`--gcp`、`--output-dir` 和 `--preview-dir`。

## 数据

训练数据与代码仓库分开维护，`train_data` 是一个私有 Git Submodule。公开用户不能直接拉取该 Submodule，需要使用有权限的账号，或准备符合相同目录和 LabelMe 格式的自有数据。

有权限的用户可以运行：

```bash
git submodule update --init train_data
```

公开用户可以将自有数据放在以下 Submodule 镜像目录中：

```text
train_data/
├── boundaries/labelme/
└── map_line_dataset/raw_data/
```

仓库中的兼容符号链接会继续向现有脚本提供：

```text
boundaries/labelme
map_line_dataset/raw_data
```

### 最小标注约定

图像与 LabelMe JSON 使用相同主文件名。JSON 需要包含 `imagePath`、`imageWidth`、`imageHeight` 和 `shapes`。当前三分类行政边界训练入口只扫描 `.jpg`，因此边界训练图像必须使用同名 `.jpg` 文件。

经纬线任务使用：

- `vertical_line`：两个点的 `line`；
- `horizontal_arc`：至少两个点的 `linestrip`；
- `splitter`：可选，用于标记需要排除的插图区域。

分类行政边界任务使用：

- `boundary_1`：一级边界 `linestrip`；
- `boundary_2`：二级边界 `linestrip`。

## 当前限制

- 各实验管线尚未收敛为统一命令行接口；
- 训练数据和模型权重不随公开仓库提供；
- 不同地图的颜色、版式、线宽和扫描质量差异较大，通常仍需人工复核；
- GCP 锚点需要人工确认，自动推断依赖经纬网格间距规则；
- 当前仓库提供 QGIS GCP 数据生成能力，不包含完整的 QGIS 工程或自动拓扑建面流程。
