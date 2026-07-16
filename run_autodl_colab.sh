#!/bin/bash
echo "=" * 60
echo "  AutoDL 训练脚本 (仿照 Colab 配置)"
echo "=" * 60
echo ""

DATA_DIR="map_line_dataset/patch_data"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
MODEL_PATH="models/unet_map_lines_autodl_colab_${TIMESTAMP}.pth"
EPOCHS=30
BATCH_SIZE=16
LR=1e-4
IMG_SIZE=512
NUM_WORKERS=2

echo "📋 训练配置"
echo "------------------------------------------------------------"
echo "   数据: $DATA_DIR"
echo "   模型: $MODEL_PATH"
echo "   Epochs: $EPOCHS"
echo "   Batch size: $BATCH_SIZE"
echo "   学习率: $LR"
echo "   图像尺寸: $IMG_SIZE"
echo "   num_workers: $NUM_WORKERS"
echo "   模式: patch_labelme (实时渲染)"
echo "   优化器: Adam"
echo "   学习率调度: CosineAnnealingLR"
echo "   类别权重: [1.0, 50.0, 50.0, 80.0]"
echo "   混合精度: AMP (CUDA)"
echo "------------------------------------------------------------"
echo ""

echo "🚀 环境检查..."
python -c "
import torch
import sys
print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU: {gpu_name} ({gpu_mem:.1f} GB)')
    print(f'CUDA: {torch.version.cuda}')
else:
    print('CUDA: 不可用')
"
echo ""

mkdir -p models
mkdir -p output

echo "🏋️  开始训练..."
echo ""

python colab_train.py \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --img_size "$IMG_SIZE" \
    --num_workers "$NUM_WORKERS"

echo ""
echo "=" * 60
echo "✅ 训练完成！"
echo "   模型保存: $MODEL_PATH"
echo "=" * 60
