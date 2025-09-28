#!/bin/bash

# Model Comparison Runner
# 模型比较运行脚本

set -e  # Exit on any error

echo "=========================================="
echo "🔬 Starting Model Comparison"
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
mkdir -p /home/yxfeng/project2/Simple-ML-project/outputs/{models,curves,reports,comparisons}
mkdir -p /home/yxfeng/project2/Simple-ML-project/logs

echo "📁 Output directories created"

# 定义要比较的模型列表
MODELS=("LogisticRegression" "RandomForest" "XGBoost" "LightGBM" "SVM" "NaiveBayes" "KNN")

echo "🔄 Running Model Comparison..."
cd /home/yxfeng/project2/Simple-ML-project

# 创建比较结果文件
COMPARISON_FILE="outputs/reports/model_comparison.csv"
echo "model_name,accuracy,precision,recall,f1_score,roc_auc,training_time" > "$COMPARISON_FILE"

# 循环运行每个模型
for model in "${MODELS[@]}"; do
    echo "🔬 Testing model: $model"
    
    # 创建临时配置文件
    TEMP_CONFIG="temp_config_${model}.yaml"
    cp "$CONFIG_FILE" "$TEMP_CONFIG"
    
    # 更新模型名称
    sed -i "s/model_name: \".*\"/model_name: \"$model\"/" "$TEMP_CONFIG"
    
    # 运行模型
    echo "🔄 Running $model..."
    python main.py \
        --data "$DATA_FILE" \
        --config "$TEMP_CONFIG" \
        --environment development \
        > "logs/${model}_comparison.log" 2>&1
    
    # 清理临时配置文件
    rm -f "$TEMP_CONFIG"
    
    echo "✅ $model completed"
done

echo "=========================================="
echo "✅ Model Comparison completed successfully!"
echo "=========================================="
echo "📊 Comparison results available in:"
echo "   - CSV Report: outputs/reports/model_comparison.csv"
echo "   - Individual Logs: logs/*_comparison.log"
echo "   - Models: outputs/models/"
echo "   - Reports: outputs/reports/"
echo "=========================================="
