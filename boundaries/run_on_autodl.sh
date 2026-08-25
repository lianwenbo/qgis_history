#!/bin/bash
# AutoDL 一键运行边界检测训练
# 在 ~/boundary_train/ 目录下执行: bash run_on_autodl.sh

set -e

echo "============================================================"
echo "  清朝地图行政边界 UNet 训练 - AutoDL"
echo "============================================================"

# 检查是否在正确目录
if [ ! -f "autodl_train_boundary.py" ]; then
    echo "ERROR: 请在 boundary_train/ 解压目录下运行此脚本"
    exit 1
fi

# 检测 Python 路径 (AutoDL 使用 miniconda3)
if command -v python &>/dev/null; then
    PY=python
    PIP=pip
elif [ -f "/root/miniconda3/bin/python" ]; then
    PY=/root/miniconda3/bin/python
    PIP=/root/miniconda3/bin/pip
elif command -v python3 &>/dev/null; then
    PY=python3
    PIP=pip3
else
    echo "ERROR: 找不到 Python"
    exit 1
fi
echo "  Python: $PY"
echo "  Pip:    $PIP"

echo ""
echo "[1/4] 检查环境..."
$PY --version
$PY -c 'import torch; print(f"  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")' 2>/dev/null || echo "  PyTorch: 未安装"

echo ""
echo "[2/4] 安装依赖..."
$PIP install -q opencv-python-headless Pillow tqdm numpy 2>&1 | tail -3

# 检查PyTorch，没有则安装CUDA版
if ! $PY -c "import torch" 2>/dev/null; then
    echo "  安装 PyTorch (CUDA 12.1)..."
    $PIP install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3
fi

$PY -c "import torch; print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo ""
echo "[3/4] 检查训练数据..."
IMG_COUNT=$(ls labelme/*.jpg 2>/dev/null | wc -l)
JSON_COUNT=$(ls labelme/*.json 2>/dev/null | wc -l)
echo "  图片: $IMG_COUNT 张, 标注: $JSON_COUNT 个"
ls -lh labelme/*.jpg

echo ""
echo "[4/4] 开始训练..."
echo "============================================================"

# 训练参数：根据GPU显存调整batch-size
#   24GB GPU (RTX3090/4090/A5000): batch_size=16
#   12GB GPU (RTX3060/4060Ti): batch_size=8
#   8GB GPU: batch_size=4
BATCH_SIZE=16
$PY -c "import torch; v=torch.cuda.get_device_properties(0).total_memory/1024**3; print(f'  GPU显存: {v:.1f} GB'); bs=16 if v>=20 else 8 if v>=10 else 4; print(f'  batch_size: {bs}')" && \
BATCH_SIZE=$($PY -c "import torch; v=torch.cuda.get_device_properties(0).total_memory/1024**3; bs=16 if v>=20 else 8 if v>=10 else 4; print(bs)")

mkdir -p models

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_PATH="models/unet_boundary_autodl_${TIMESTAMP}.pth"
echo "  模型输出: $MODEL_PATH"

$PY autodl_train_boundary.py \
    --labelme_dir labelme \
    --model_path "$MODEL_PATH" \
    --epochs 40 \
    --batch_size $BATCH_SIZE \
    --lr 2e-4 \
    --patch_size 512 \
    --overlap 0.5 \
    --bg_ratio 0.2 \
    --line_thickness_1 5 \
    --line_thickness_2 2 \
    --num_workers 4

echo ""
echo "============================================================"
echo "  训练完成！"
echo "  模型文件: $(pwd)/$MODEL_PATH"
echo ""
echo "  下载到本地:"
echo "  scp -P 50188 root@connect.westc.seetacloud.com:$(pwd)/$MODEL_PATH ~/Downloads/"
echo "============================================================"
