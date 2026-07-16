#!/bin/bash
echo "=" * 60
echo "  GPU 实时监控"
echo "=" * 60
echo ""
echo "按 Ctrl+C 停止监控"
echo ""

while true; do
    clear
    echo "=" * 60
    echo "  GPU / CPU 实时监控"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=" * 60
    echo ""
    
    echo "--- GPU 状态 ---"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi 不可用)"
    
    echo ""
    echo "--- GPU 进程 ---"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || echo "  (无进程)"
    
    echo ""
    echo "--- Python 进程 ---"
    ps aux | grep python | grep -v grep | head -5
    
    echo ""
    echo "=" * 60
    sleep 2
done
