#!/bin/bash
# 打包边界训练数据+脚本（基于git archive，只打包已提交版本）
# 用法: bash boundaries/pack_boundary_train.sh
# 输出: /tmp/boundary_train.tar.gz

set -e

cd "$(dirname "$0")/.."
PROJ_ROOT=$(pwd)

echo "=== 打包边界训练数据 (git archive) ==="
echo "项目根目录: $PROJ_ROOT"

# 检查工作区是否干净（boundaries 相关文件）
echo ""
echo "[1/3] 检查工作区状态..."
DIRTY=$(git status --porcelain boundaries/ | grep -v '^??' || true)
UNTRACKED=$(git status --porcelain boundaries/ | grep '^??' || true)

if [ -n "$DIRTY" ]; then
    echo "⚠️  boundaries/ 下有未提交的修改:"
    echo "$DIRTY"
    echo ""
    echo "请先 commit 后再打包。"
    echo "  git add boundaries/ && git commit -m 'your message'"
    exit 1
fi

if [ -n "$UNTRACKED" ]; then
    echo "⚠️  boundaries/ 下有未跟踪的新文件:"
    echo "$UNTRACKED"
    echo ""
    echo "请先 git add && git commit 后再打包。"
    exit 1
fi

GIT_SHA=$(git rev-parse --short HEAD)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "  Git SHA: $GIT_SHA"
echo "  工作区干净，使用 git archive 打包"

# 临时目录
STAGE_DIR=$(mktemp -d)
TRAIN_DIR="$STAGE_DIR/boundary_train"
mkdir -p "$TRAIN_DIR"

echo ""
echo "[2/3] 从 git 提取文件..."

# 用 git archive 提取 boundaries/ 目录，strip 掉 boundaries/ 前缀
git archive HEAD boundaries/ | tar -xf - -C "$STAGE_DIR" --strip-components=1

# 移动需要的文件到 TRAIN_DIR，清理不需要的
cd "$STAGE_DIR"

# 需要保留的文件/目录
KEEP=(
    "autodl_train_boundary.py"
    "run_on_autodl.sh"
    "labelme"
)

mkdir -p "$TRAIN_DIR"
for item in "${KEEP[@]}"; do
    if [ -e "$item" ]; then
        mv "$item" "$TRAIN_DIR/"
        echo "  ✓ $item"
    else
        echo "  ✗ $item (不存在，跳过)"
    fi
done

# 确保run_on_autodl.sh可执行
chmod +x "$TRAIN_DIR/run_on_autodl.sh"

# 统计
IMG_COUNT=$(ls "$TRAIN_DIR/labelme/"*.jpg 2>/dev/null | wc -l)
echo ""
echo "  训练图片: $IMG_COUNT 张"
ls "$TRAIN_DIR/labelme/"*.jpg 2>/dev/null | while read f; do echo "    $(basename "$f")"; done

# 打包
OUT_TAR="/tmp/boundary_train.tar.gz"
rm -f "$OUT_TAR"
echo ""
echo "[3/3] 打包: $OUT_TAR"
cd "$STAGE_DIR"
tar -czf "$OUT_TAR" boundary_train/

# 清理
rm -rf "$STAGE_DIR"

SIZE=$(du -h "$OUT_TAR" | cut -f1)
echo ""
echo "=== 打包完成 ==="
echo "文件: $OUT_TAR"
echo "大小: $SIZE"
echo ""
echo "上传到 AutoDL (西部区):"
echo "  scp -P 50188 $OUT_TAR root@connect.westc.seetacloud.com:/root/"
echo "  ssh -p 50188 root@connect.westc.seetacloud.com"
echo "  cd ~ && tar -xzf boundary_train.tar.gz && cd boundary_train"
echo "  bash run_on_autodl.sh"
