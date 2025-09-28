#!/bin/bash

# Production ML Pipeline Runner
# 生产环境机器学习管道运行脚本

set -e  # Exit on any error

echo "=========================================="
echo "🏭 Starting Production ML Pipeline"
echo "=========================================="

# 设置生产环境变量
export PYTHONPATH="/home/yxfeng/project2/Simple-ML-project/src:$PYTHONPATH"
export ML_CONFIG_ENVIRONMENT="production"
export ML_CONFIG_CACHE_USE_DISK_CACHE="true"
export ML_CONFIG_LOGGER_LEVEL="INFO"
export ML_CONFIG_MONITORING_INTERVAL="10.0"

# 检查必要文件
DATA_FILE="/home/yxfeng/project2/Simple-ML-project/data/train_u6lujuX_CVtuZ9i.csv"
CONFIG_FILE="/home/yxfeng/project2/Simple-ML-project/config.yaml"

if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Error: Data file not found at $DATA_FILE"
    echo "Please ensure the data file exists or update the path in this script"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

echo "✅ Data file found: $DATA_FILE"
echo "✅ Config file found: $CONFIG_FILE"

# 创建生产环境输出目录
mkdir -p /home/yxfeng/project2/Simple-ML-project/outputs/{models,curves,reports}
mkdir -p /home/yxfeng/project2/Simple-ML-project/logs
mkdir -p /home/yxfeng/project2/Simple-ML-project/cache
mkdir -p /home/yxfeng/project2/Simple-ML-project/swanlog

echo "📁 Production output directories created"

# 设置日志文件权限
chmod 755 /home/yxfeng/project2/Simple-ML-project/logs

# 运行生产环境主程序
echo "🔄 Running Production ML Pipeline..."
cd /home/yxfeng/project2/Simple-ML-project

# 使用nohup在后台运行，并记录日志
nohup python main.py \
    --data "$DATA_FILE" \
    --config "$CONFIG_FILE" \
    --environment production \
    --async-mode \
    > logs/production_pipeline.log 2>&1 &

PIPELINE_PID=$!
echo "🔄 Pipeline started with PID: $PIPELINE_PID"

# 等待一段时间让管道启动
sleep 5

# 检查进程是否还在运行
if ps -p $PIPELINE_PID > /dev/null; then
    echo "✅ Production pipeline is running successfully"
    echo "📊 Monitor the logs with: tail -f logs/production_pipeline.log"
    echo "🛑 To stop the pipeline: kill $PIPELINE_PID"
else
    echo "❌ Pipeline failed to start. Check logs/production_pipeline.log for details"
    exit 1
fi

echo "=========================================="
echo "🏭 Production ML Pipeline is running!"
echo "=========================================="
echo "📊 Monitor the pipeline:"
echo "   - Logs: tail -f logs/production_pipeline.log"
echo "   - Reports: outputs/reports/"
echo "   - Models: outputs/models/"
echo "   - PID: $PIPELINE_PID"
echo "=========================================="
