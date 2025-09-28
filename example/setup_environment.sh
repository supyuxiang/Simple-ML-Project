#!/bin/bash

# Environment Setup Script
# 环境设置脚本

set -e  # Exit on any error

echo "=========================================="
echo "🛠️  Setting up ML Project Environment"
echo "=========================================="

# 检查Python版本
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $python_version"

# 检查pip
echo "📦 Checking pip..."
pip --version

# 创建虚拟环境（可选）
read -p "🤔 Do you want to create a virtual environment? (y/n): " create_venv
if [ "$create_venv" = "y" ] || [ "$create_venv" = "Y" ]; then
    echo "🔄 Creating virtual environment..."
    python3 -m venv /home/yxfeng/project2/Simple-ML-project/venv
    source /home/yxfeng/project2/Simple-ML-project/venv/bin/activate
    echo "✅ Virtual environment created and activated"
fi

# 安装依赖
echo "📦 Installing dependencies..."
cd /home/yxfeng/project2/Simple-ML-project

# 检查requirements.txt
if [ -f "requirements.txt" ]; then
    echo "📋 Installing from requirements.txt..."
    pip install -r requirements.txt
else
    echo "📋 Installing core dependencies..."
    pip install numpy pandas scikit-learn matplotlib seaborn
    pip install xgboost lightgbm
    pip install pyyaml
    pip install psutil
    pip install cryptography
    pip install swanlab  # 可选，用于实验跟踪
fi

# 创建必要的目录
echo "📁 Creating directories..."
mkdir -p outputs/{models,curves,reports,comparisons}
mkdir -p logs
mkdir -p cache
mkdir -p swanlog
mkdir -p test_outputs

echo "✅ Directories created"

# 设置权限
echo "🔐 Setting permissions..."
chmod +x example/*.sh
chmod 755 logs
chmod 755 cache

echo "✅ Permissions set"

# 检查数据文件
echo "📊 Checking data files..."
DATA_DIR="/home/yxfeng/project2/Simple-ML-project/data"
if [ -d "$DATA_DIR" ]; then
    echo "✅ Data directory found: $DATA_DIR"
    ls -la "$DATA_DIR"
else
    echo "⚠️  Data directory not found: $DATA_DIR"
    echo "Please ensure your data files are in the correct location"
fi

# 运行基本测试
echo "🧪 Running basic tests..."
python -c "
import sys
sys.path.insert(0, 'src')
try:
    import src.core
    import src.data
    import src.models
    import src.training
    import src.evaluation
    print('✅ All modules import successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

echo "=========================================="
echo "✅ Environment setup completed successfully!"
echo "=========================================="
echo "🚀 You can now run the ML pipeline using:"
echo "   - Basic: ./example/run_basic_pipeline.sh"
echo "   - Async: ./example/run_async_pipeline.sh"
echo "   - Production: ./example/run_production_pipeline.sh"
echo "   - Tests: ./example/run_tests.sh"
echo "   - Model Comparison: ./example/run_model_comparison.sh"
echo "=========================================="
