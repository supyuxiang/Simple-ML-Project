#!/bin/bash

# 朴素贝叶斯模型运行脚本
# Naive Bayes Model Pipeline

echo "📊 启动朴素贝叶斯模型管道..."
echo "=================================="

# 设置环境
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CONDA_DEFAULT_ENV="fyx_sci"

# 数据文件路径
DATA_FILE="data/train_u6lujuX_CVtuZ9i.csv"

# 检查数据文件是否存在
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ 错误: 数据文件 $DATA_FILE 不存在"
    exit 1
fi

echo "📊 使用数据文件: $DATA_FILE"
echo "📊 使用模型: NaiveBayes"
echo ""

# 创建临时配置文件
TEMP_CONFIG="temp_naive_bayes_config.yaml"
cat > "$TEMP_CONFIG" << EOF
Model:
  model_name: "NaiveBayes"
  model_type: "classification"
  model_params:
    var_smoothing: 0.000000001

Data:
  data_file: "$DATA_FILE"
  target_column: "Loan_Status"
  test_size: 0.2
  random_state: 42
  categorical_columns:
    - "Gender"
    - "Married"
    - "Dependents"
    - "Education"
    - "Self_Employed"
    - "Property_Area"

Training:
  cv_folds: 5
  random_state: 42
  early_stopping_rounds: 10
  validation_split: 0.2

Logging:
  level: "INFO"
  log_file: "logs/naive_bayes.log"
  console_output: true

Cache:
  enabled: true
  backend: "memory"
  ttl: 3600

Monitoring:
  enabled: true
  metrics_interval: 10
EOF

echo "⚙️  使用临时配置文件: $TEMP_CONFIG"
echo ""

# 运行管道
echo "🏃 开始训练朴素贝叶斯模型..."
python main.py --data "$DATA_FILE" --config "$TEMP_CONFIG"

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 朴素贝叶斯模型训练完成！"
    echo "📁 检查以下文件获取结果:"
    echo "   - logs/naive_bayes.log (训练日志)"
    echo "   - results/ (模型结果和报告)"
    echo "   - models/ (保存的模型文件)"
else
    echo ""
    echo "❌ 朴素贝叶斯模型训练失败！"
    echo "📋 请检查日志文件获取详细错误信息"
fi

# 清理临时文件
rm -f "$TEMP_CONFIG"

echo ""
echo "🏁 朴素贝叶斯模型管道执行完毕"
