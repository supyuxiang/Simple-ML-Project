#!/bin/bash

# Pipeline Monitor
# 管道监控脚本

echo "=========================================="
echo "📊 ML Pipeline Monitor"
echo "=========================================="

# 检查运行中的进程
echo "🔄 Checking running processes..."
PIPELINE_PIDS=$(pgrep -f "python main.py" || echo "")
if [ -n "$PIPELINE_PIDS" ]; then
    echo "✅ Found running pipeline processes:"
    ps -p $PIPELINE_PIDS -o pid,ppid,cmd,etime,pcpu,pmem
else
    echo "❌ No pipeline processes found"
fi

# 检查日志文件
echo ""
echo "📋 Recent log entries:"
LOG_FILE="/home/yxfeng/project2/Simple-ML-project/logs"
if [ -d "$LOG_FILE" ]; then
    echo "📁 Log directory: $LOG_FILE"
    find "$LOG_FILE" -name "*.log" -type f -exec ls -la {} \; | head -10
    
    echo ""
    echo "📄 Recent log content:"
    find "$LOG_FILE" -name "*.log" -type f -exec tail -5 {} \; 2>/dev/null | head -20
else
    echo "❌ Log directory not found"
fi

# 检查输出文件
echo ""
echo "📊 Output files:"
OUTPUT_DIR="/home/yxfeng/project2/Simple-ML-project/outputs"
if [ -d "$OUTPUT_DIR" ]; then
    echo "📁 Output directory: $OUTPUT_DIR"
    find "$OUTPUT_DIR" -type f -exec ls -la {} \; | head -10
else
    echo "❌ Output directory not found"
fi

# 检查系统资源
echo ""
echo "💻 System resources:"
echo "CPU usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
echo "Memory usage:"
free -h | grep "Mem:" | awk '{print "Used: " $3 " / " $2 " (" $3/$2*100 "%)"}'
echo "Disk usage:"
df -h /home/yxfeng/project2/Simple-ML-project | tail -1 | awk '{print "Used: " $3 " / " $2 " (" $5 ")"}'

# 检查缓存
echo ""
echo "🗄️ Cache status:"
CACHE_DIR="/home/yxfeng/project2/Simple-ML-project/cache"
if [ -d "$CACHE_DIR" ]; then
    CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
    CACHE_FILES=$(find "$CACHE_DIR" -type f | wc -l)
    echo "📁 Cache directory: $CACHE_DIR"
    echo "📊 Cache size: $CACHE_SIZE"
    echo "📄 Cache files: $CACHE_FILES"
else
    echo "❌ Cache directory not found"
fi

# 提供操作选项
echo ""
echo "=========================================="
echo "🛠️  Available operations:"
echo "=========================================="
echo "1. View live logs: tail -f logs/*.log"
echo "2. Stop pipeline: pkill -f 'python main.py'"
echo "3. Clean cache: rm -rf cache/*"
echo "4. Clean logs: rm -rf logs/*"
echo "5. Clean outputs: rm -rf outputs/*"
echo "6. Restart pipeline: ./example/run_basic_pipeline.sh"
echo "=========================================="

# 交互式选项
read -p "🤔 Do you want to perform any operations? (1-6 or 'q' to quit): " choice

case $choice in
    1)
        echo "📄 Showing live logs (Ctrl+C to exit):"
        tail -f /home/yxfeng/project2/Simple-ML-project/logs/*.log 2>/dev/null || echo "No log files found"
        ;;
    2)
        echo "🛑 Stopping pipeline processes..."
        pkill -f "python main.py" && echo "✅ Pipeline stopped" || echo "❌ No processes to stop"
        ;;
    3)
        echo "🗑️ Cleaning cache..."
        rm -rf /home/yxfeng/project2/Simple-ML-project/cache/* && echo "✅ Cache cleaned" || echo "❌ Cache cleaning failed"
        ;;
    4)
        echo "🗑️ Cleaning logs..."
        rm -rf /home/yxfeng/project2/Simple-ML-project/logs/* && echo "✅ Logs cleaned" || echo "❌ Log cleaning failed"
        ;;
    5)
        echo "🗑️ Cleaning outputs..."
        rm -rf /home/yxfeng/project2/Simple-ML-project/outputs/* && echo "✅ Outputs cleaned" || echo "❌ Output cleaning failed"
        ;;
    6)
        echo "🔄 Restarting pipeline..."
        /home/yxfeng/project2/Simple-ML-project/example/run_basic_pipeline.sh
        ;;
    q|Q)
        echo "👋 Goodbye!"
        ;;
    *)
        echo "❌ Invalid option"
        ;;
esac

echo "=========================================="
echo "✅ Monitor session completed"
echo "=========================================="
