#!/bin/bash

# 所有模型比较脚本
# All Models Comparison Pipeline

echo "🏆 启动所有模型比较管道..."
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
echo "🏆 将比较所有可用模型"
echo ""

# 创建结果目录
mkdir -p results/model_comparison
mkdir -p logs/model_comparison

# 模型列表
MODELS=("LogisticRegression" "RandomForest" "XGBoost" "LightGBM" "SVM" "NaiveBayes" "KNN")

# 存储结果的数组
declare -a RESULTS=()

echo "🚀 开始训练所有模型..."
echo "=================================="

# 遍历所有模型
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "🤖 训练模型: $MODEL"
    echo "--------------------------------"
    
    # 创建临时配置文件
    TEMP_CONFIG="temp_${MODEL,,}_config.yaml"
    
    case $MODEL in
        "LogisticRegression")
            cat > "$TEMP_CONFIG" << EOF
Model:
  model_name: "LogisticRegression"
  model_type: "classification"
  model_params:
    random_state: 42
    max_iter: 1000
    C: 1.0
    solver: "liblinear"
EOF
            ;;
        "RandomForest")
            cat > "$TEMP_CONFIG" << EOF
Model:
  model_name: "RandomForest"
  model_type: "classification"
  model_params:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
    min_samples_leaf: 2
    random_state: 42
    n_jobs: -1
EOF
            ;;
        "XGBoost")
            cat > "$TEMP_CONFIG" << EOF
Model:
  model_name: "XGBoost"
  model_type: "classification"
  model_params:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 6
    random_state: 42
EOF
            ;;
        "LightGBM")
            cat > "$TEMP_CONFIG" << EOF
Model:
  model_name: "LightGBM"
  model_type: "classification"
  model_params:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 6
    num_leaves: 31
    feature_fraction: 0.9
    bagging_fraction: 0.8
    bagging_freq: 5
    random_state: 42
    verbose: -1
EOF
            ;;
        "SVM")
            cat > "$TEMP_CONFIG" << EOF
Model:
  model_name: "SVM"
  model_type: "classification"
  model_params:
    random_state: 42
    probability: true
    kernel: "rbf"
    C: 1.0
    gamma: "scale"
EOF
            ;;
        "NaiveBayes")
            cat > "$TEMP_CONFIG" << EOF
Model:
  model_name: "NaiveBayes"
  model_type: "classification"
  model_params:
    var_smoothing: 1e-9
EOF
            ;;
        "KNN")
            cat > "$TEMP_CONFIG" << EOF
Model:
  model_name: "KNN"
  model_type: "classification"
  model_params:
    n_neighbors: 5
    weights: "uniform"
    algorithm: "auto"
    leaf_size: 30
    p: 2
    metric: "minkowski"
EOF
            ;;
    esac
    
    # 添加通用配置
    cat >> "$TEMP_CONFIG" << EOF

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
  log_file: "logs/model_comparison/${MODEL,,}.log"
  console_output: false

Cache:
  enabled: true
  backend: "memory"
  ttl: 3600

Monitoring:
  enabled: true
  metrics_interval: 10
EOF
    
    # 运行模型训练
    echo "🏃 开始训练 $MODEL..."
    START_TIME=$(date +%s)
    
    python main.py --data "$DATA_FILE" --config "$TEMP_CONFIG" > "logs/model_comparison/${MODEL,,}_output.log" 2>&1
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    # 检查运行结果
    if [ $? -eq 0 ]; then
        echo "✅ $MODEL 训练完成 (耗时: ${DURATION}秒)"
        RESULTS+=("$MODEL:SUCCESS:${DURATION}s")
    else
        echo "❌ $MODEL 训练失败 (耗时: ${DURATION}秒)"
        RESULTS+=("$MODEL:FAILED:${DURATION}s")
    fi
    
    # 清理临时文件
    rm -f "$TEMP_CONFIG"
done

echo ""
echo "=================================="
echo "🏆 所有模型训练完成！"
echo "=================================="

# 显示结果总结
echo ""
echo "📊 训练结果总结:"
echo "--------------------------------"
for RESULT in "${RESULTS[@]}"; do
    IFS=':' read -r MODEL STATUS DURATION <<< "$RESULT"
    if [ "$STATUS" = "SUCCESS" ]; then
        echo "✅ $MODEL - 成功 ($DURATION)"
    else
        echo "❌ $MODEL - 失败 ($DURATION)"
    fi
done

echo ""
echo "📁 结果文件位置:"
echo "   - 日志文件: logs/model_comparison/"
echo "   - 模型文件: models/"
echo "   - 结果报告: results/"
echo ""
echo "🔍 查看详细结果:"
echo "   - 训练日志: ls logs/model_comparison/"
echo "   - 模型性能: 检查 results/ 目录中的报告文件"
echo ""
echo "🏁 模型比较管道执行完毕"
