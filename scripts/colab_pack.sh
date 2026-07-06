#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_ZIP="$PROJECT_DIR/colab_train.zip"

echo "📦 打包 Colab 训练文件..."
echo "项目目录: $PROJECT_DIR"
echo "输出文件: $OUTPUT_ZIP"

cd "$PROJECT_DIR"

rm -f "$OUTPUT_ZIP"

zip -r "$OUTPUT_ZIP" \
  unet_model.py \
  unet_data_prep.py \
  post_processing.py \
  map_line_dataset/patch_data/ \
  --exclude "*.DS_Store" \
  --exclude "*/__pycache__/*"

echo ""
echo "✅ 打包完成！"
echo "输出: $OUTPUT_ZIP"
SIZE=$(du -h "$OUTPUT_ZIP" | cut -f1)
echo "大小: $SIZE"
echo ""
echo "使用方法:"
echo "  1. 上传 colab_train.zip 到 Google Drive 或 Colab 会话"
echo "  2. 在 Colab 中运行: !unzip colab_train.zip"
echo "  3. 运行训练: !python colab_train.py"
