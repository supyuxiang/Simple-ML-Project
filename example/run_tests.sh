#!/bin/bash

# Test Suite Runner
# 测试套件运行脚本

set -e  # Exit on any error

echo "=========================================="
echo "🧪 Starting Test Suite"
echo "=========================================="

# 设置环境变量
export PYTHONPATH="/home/yxfeng/project2/Simple-ML-project/src:$PYTHONPATH"
export ML_CONFIG_ENVIRONMENT="test"

# 检查必要文件
CONFIG_FILE="/home/yxfeng/project2/Simple-ML-project/config.yaml"
TEST_FILE="/home/yxfeng/project2/Simple-ML-project/test_production_system.py"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file not found at $TEST_FILE"
    exit 1
fi

echo "✅ Config file found: $CONFIG_FILE"
echo "✅ Test file found: $TEST_FILE"

# 创建测试输出目录
mkdir -p /home/yxfeng/project2/Simple-ML-project/test_outputs
mkdir -p /home/yxfeng/project2/Simple-ML-project/logs

echo "📁 Test output directories created"

# 运行测试
echo "🔄 Running Test Suite..."
cd /home/yxfeng/project2/Simple-ML-project

# 运行生产系统测试
echo "🧪 Running Production System Tests..."
python test_production_system.py

# 运行单元测试（如果存在）
if [ -d "tests" ]; then
    echo "🧪 Running Unit Tests..."
    python -m pytest tests/ -v --tb=short
fi

echo "=========================================="
echo "✅ Test Suite completed successfully!"
echo "=========================================="
echo "📊 Test results available in:"
echo "   - Test outputs: test_outputs/"
echo "   - Logs: logs/"
echo "=========================================="
