#!/bin/bash
# qgis_only 项目环境激活脚本

# 设置项目根目录
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 激活conda环境
echo "🔧 激活 qgis 环境..."
source /opt/homebrew/anaconda3/etc/profile.d/conda.sh 2>/dev/null || {
    echo "⚠️  conda初始化脚本未找到，尝试直接激活..."
}

conda activate qgis 2>/dev/null || {
    echo "❌ 无法激活 qgis 环境"
    echo "请确保已创建环境: conda create -n qgis python=3.12"
    return 1
}

# 确保项目目录的Python优先
export PATH="/opt/homebrew/anaconda3/envs/qgis/bin:$PATH"

# 设置项目别名
alias py="python"
alias pip="python -m pip"

# 打印环境信息
echo ""
echo "✅ qgis 环境已激活!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📍 项目目录: $PROJECT_DIR"
echo "🐍 Python路径: $(which python)"
echo "📦 Python版本: $(python --version 2>&1)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 切换到项目目录
cd "$PROJECT_DIR"
