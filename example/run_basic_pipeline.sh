#!/bin/bash

# Basic ML Pipeline Runner
# 基础机器学习管道运行脚本

set -e  # Exit on any error

echo "=========================================="
echo "🚀 Starting Basic ML Pipeline"
echo "=========================================="

# 设置环境变量
export PYTHONPATH="/home/yxfeng/project2/Simple-ML-project/src:$PYTHONPATH"
export ML_CONFIG_ENVIRONMENT="development"

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

# 创建输出目录
mkdir -p /home/yxfeng/project2/Simple-ML-project/outputs/{models,curves,reports}
mkdir -p /home/yxfeng/project2/Simple-ML-project/logs
mkdir -p /home/yxfeng/project2/Simple-ML-project/cache

echo "📁 Output directories created"

# 运行主程序
echo "🔄 Running ML Pipeline..."
cd /home/yxfeng/project2/Simple-ML-project

python main.py \
    --data "$DATA_FILE" \
    --config "$CONFIG_FILE" \
    --environment development

echo "=========================================="
echo "✅ Basic ML Pipeline completed successfully!"
echo "=========================================="
echo "📊 Check the outputs directory for results:"
echo "   - Models: outputs/models/"
echo "   - Reports: outputs/reports/"
echo "   - Logs: logs/"
echo "=========================================="
