# ML1 - Loan Prediction System

A comprehensive, enterprise-grade machine learning pipeline for loan approval prediction, built with modern software engineering practices and advanced ML techniques.

## 🚀 Features

### Core Capabilities
- **Multi-Model Support**: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, Naive Bayes, KNN
- **Advanced Feature Engineering**: Polynomial features, clustering features, interaction features, PCA
- **Comprehensive Data Processing**: Missing value handling, outlier detection, data validation
- **Enterprise Architecture**: Modular design, configuration management, logging system
- **Model Evaluation**: Cross-validation, comprehensive metrics, visualization
- **Experiment Tracking**: SwanLab integration for experiment management

### Technical Highlights
- **OOP Design**: Clean, maintainable, and extensible codebase
- **Configuration-Driven**: YAML-based configuration management
- **Comprehensive Logging**: Multi-level logging with experiment tracking
- **Data Validation**: Automated data quality checks and recommendations
- **Feature Engineering**: Advanced feature creation and selection
- **Model Persistence**: Save/load trained models with metadata
- **Visualization**: Rich plots and evaluation reports

## 📁 Project Structure

```
ML1/
├── src/                          # Source code
│   ├── core/                     # Core base classes and interfaces
│   │   ├── base.py              # Abstract base classes
│   │   ├── config.py            # Configuration management
│   │   └── logger.py            # Logging system
│   ├── data/                     # Data processing modules
│   │   ├── processor.py         # Main data processor
│   │   ├── feature_engineering.py # Feature engineering
│   │   └── validation.py        # Data validation
│   ├── models/                   # Machine learning models
│   │   ├── base_model.py        # Base model class
│   │   └── sklearn_models.py    # Sklearn model implementations
│   ├── training/                 # Training modules
│   │   └── trainer.py           # Model trainer
│   ├── evaluation/               # Evaluation modules
│   │   └── evaluator.py         # Model evaluator
│   └── utils/                    # Utility functions
│       └── helpers.py           # Helper functions
├── data/                         # Data directory
│   └── train_u6lujuX_CVtuZ9i.csv # Training data
├── outputs/                      # Output directory
│   ├── models/                   # Saved models
│   ├── curves/                   # Plots and visualizations
│   └── reports/                  # Evaluation reports
├── logs/                         # Log files
├── config.yaml                   # Configuration file
├── main.py                       # Main entry point
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd ML1
```

2. **Create virtual environment**
```bash
# Using conda (recommended)
conda create -n ml1 python=3.9
conda activate ml1

# Or using venv
python -m venv ml1_env
source ml1_env/bin/activate  # On Windows: ml1_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import pandas, sklearn, xgboost, lightgbm; print('Installation successful!')"
```

## 🚀 Quick Start

### Basic Usage

1. **Run the complete pipeline**
```bash
python main.py --data data/train_u6lujuX_CVtuZ9i.csv --config config.yaml
```

2. **Train specific model**
```bash
# Edit config.yaml to set model_name to your preferred model
python main.py --data data/train_u6lujuX_CVtuZ9i.csv
```

3. **Run with custom configuration**
```bash
python main.py --data data/train_u6lujuX_CVtuZ9i.csv --config my_config.yaml
```

### Programmatic Usage

```python
from src.core import ConfigManager, Logger
from src.data import LoanDataProcessor
from src.models import RandomForestModel
from src.training import ModelTrainer
from src.evaluation import ModelEvaluator

# Initialize components
config_manager = ConfigManager("config.yaml")
logger = Logger(config_manager.get_logger_config())

# Load and preprocess data
processor = LoanDataProcessor(config_manager.get_data_config(), logger)
df = processor.load_data("data/train_u6lujuX_CVtuZ9i.csv")
X, y = processor.preprocess(df)
X_train, X_val, y_train, y_val = processor.split_data(X, y)

# Train model
model = RandomForestModel(config_manager.get_model_config(), logger)
trainer = ModelTrainer(config_manager.get_train_config(), logger)
trainer.set_model(model)
trainer.train(X_train, y_train, X_val, y_val)

# Evaluate model
evaluator = ModelEvaluator(config_manager.get_metrics_config(), logger)
predictions = model.predict(X_val)
probabilities = model.predict_proba(X_val)
metrics = evaluator.compute_metrics(y_val, predictions, probabilities)

print(f"Model accuracy: {metrics['accuracy']:.4f}")
```

## ⚙️ Configuration

The system is fully configurable through `config.yaml`. Key configuration sections:

### Model Configuration
```yaml
Model:
  model_name: "RandomForest"  # or "XGBoost", "LightGBM", etc.
  model_type: "classification"
  model_params:
    n_estimators: 100
    max_depth: 10
    random_state: 42
```

### Data Processing
```yaml
Data:
  test_size: 0.2
  random_seed: 42
  missing_strategy: "smart"  # "smart", "drop", "fill"
  scale_features: true
  scaling_method: "standard"  # "standard", "minmax", "none"
```

### Training Configuration
```yaml
Train:
  epochs: 100
  batch_size: 8
  learning_rate: 0.01
  cv_folds: 5
  early_stopping: 10
```

### Evaluation Metrics
```yaml
Metrics:
  metrics_list:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1_score"
    - "roc_auc"
    - "confusion_matrix"
```

## 📊 Data Format

The system expects CSV data with the following structure:

| Column | Type | Description |
|--------|------|-------------|
| Loan_ID | String | Unique loan identifier |
| Gender | Categorical | Male/Female |
| Married | Categorical | Yes/No |
| Dependents | Categorical | Number of dependents |
| Education | Categorical | Graduate/Not Graduate |
| Self_Employed | Categorical | Yes/No |
| ApplicantIncome | Numeric | Applicant's income |
| CoapplicantIncome | Numeric | Co-applicant's income |
| LoanAmount | Numeric | Loan amount requested |
| Loan_Amount_Term | Numeric | Loan term in months |
| Credit_History | Numeric | Credit history (0/1) |
| Property_Area | Categorical | Urban/Rural/Semiurban |
| Loan_Status | Categorical | Y/N (target variable) |

## 🎯 Model Performance

The system supports multiple models with different strengths:

| Model | Accuracy | F1-Score | Training Time | Best For |
|-------|----------|----------|---------------|----------|
| Random Forest | ~0.85 | ~0.84 | Fast | General purpose |
| XGBoost | ~0.87 | ~0.86 | Medium | High performance |
| LightGBM | ~0.86 | ~0.85 | Fast | Large datasets |
| Logistic Regression | ~0.82 | ~0.81 | Very Fast | Interpretability |
| SVM | ~0.83 | ~0.82 | Slow | Small datasets |

*Performance may vary based on data and configuration*

## 🔧 Advanced Features

### Feature Engineering
- **Polynomial Features**: Create interaction terms
- **Clustering Features**: Add cluster-based features
- **PCA**: Dimensionality reduction
- **Feature Selection**: Automatic feature importance selection

### Data Validation
- **Missing Value Analysis**: Comprehensive missing data assessment
- **Outlier Detection**: Multiple outlier detection methods
- **Data Consistency**: Logical constraint validation
- **Quality Scoring**: Automated data quality assessment

### Model Evaluation
- **Cross-Validation**: K-fold cross-validation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Visualization**: Confusion matrix, ROC curves, feature importance
- **Model Comparison**: Side-by-side model performance comparison

## 📈 Outputs

The system generates comprehensive outputs:

### Models
- Trained model files (`.pkl` format)
- Model metadata and parameters
- Feature importance scores

### Visualizations
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Feature importance plots
- Model comparison charts

### Reports
- Evaluation metrics summary
- Data quality report
- Model performance comparison
- Training history logs

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📝 Logging

The system provides comprehensive logging:

- **Console Logging**: Real-time progress updates
- **File Logging**: Detailed logs saved to files
- **Experiment Tracking**: SwanLab integration for experiment management
- **Structured Logging**: JSON-formatted logs for analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 mypy pytest

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Scikit-learn team for the excellent ML library
- XGBoost and LightGBM teams for gradient boosting libraries
- Pandas and NumPy teams for data manipulation tools
- Matplotlib and Seaborn teams for visualization tools

## 📞 Support

For questions, issues, or contributions:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed description
3. Contact the maintainers

---

**Built with ❤️ for the machine learning community**
