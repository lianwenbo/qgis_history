#!/bin/bash
if [ -z "$1" ]; then
    echo "用法: $0 <图像路径> [模型路径]"
    echo "示例: $0 map_line_dataset/raw_data/05-44河南道.jpg"
    exit 1
fi

IMAGE_PATH="$1"
MODEL_PATH="${2:-models/unet_map_lines_autodl_colab.pth}"
OUTPUT_DIR="output"

echo "=" * 60
echo "  UNet 推理"
echo "=" * 60
echo ""
echo "模型: $MODEL_PATH"
echo "输入: $IMAGE_PATH"
echo "输出: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

python -c "
import sys
sys.path.insert(0, '.')
from unet_model import predict_unet_sliding_window

predict_unet_sliding_window(
    model_path='$MODEL_PATH',
    image_path='$IMAGE_PATH',
    output_dir='$OUTPUT_DIR',
    patch_size=512,
    overlap=0.5
)
"

echo ""
echo "✅ 推理完成！"
echo "结果保存在: $OUTPUT_DIR/"
