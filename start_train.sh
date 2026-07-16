#!/bin/bash
echo "=" * 60
echo "  后台启动训练 (AutoDL Colab 配置)"
echo "=" * 60
echo ""

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo "日志文件: $LOG_FILE"
echo ""

nohup bash run_autodl_colab.sh > "$LOG_FILE" 2>&1 &

PID=$!
echo "训练进程 PID: $PID"
echo ""

echo "查看进度:"
echo "  tail -f $LOG_FILE"
echo ""
echo "查看 GPU:"
echo "  bash monitor_gpu.sh"
echo ""
echo "停止训练:"
echo "  kill $PID"
echo ""
echo "=" * 60
