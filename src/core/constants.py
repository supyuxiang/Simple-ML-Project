"""
Constants and configuration defaults for the ML project.
This module contains all the constant values used throughout the project.
"""

from typing import Dict, List, Any
from enum import Enum


class ModelType(Enum):
    """Supported model types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ScalingMethod(Enum):
    """Supported scaling methods."""
    STANDARD = "standard"
    MINMAX = "minmax"
    NONE = "none"


class MissingStrategy(Enum):
    """Supported missing value handling strategies."""
    DROP = "drop"
    FILL = "fill"
    SMART = "smart"


class ImputationMethod(Enum):
    """Supported imputation methods."""
    MEAN = "mean"
    MEDIAN = "median"
    MOST_FREQUENT = "most_frequent"
    KNN = "knn"


# Default configuration values
DEFAULT_CONFIG: Dict[str, Any] = {
    "Model": {
        "model_name": "XGBoost",
        "model_type": ModelType.CLASSIFICATION.value,
        "model_params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "random_state": 42
        }
    },
    "Data": {
        "test_size": 0.2,
        "validation_size": 0.2,
        "random_seed": 42,
        "missing_strategy": MissingStrategy.SMART.value,
        "categorical_imputer": ImputationMethod.MOST_FREQUENT.value,
        "numerical_imputer": ImputationMethod.MEDIAN.value,
        "scale_features": True,
        "scaling_method": ScalingMethod.STANDARD.value,
        "create_features": True,
        "feature_selection": True,
        "n_features_select": 10
    },
    "Train": {
        "cv_folds": 5,
        "random_seed": 42,
        "verbose": 1,
        "early_stopping": 10,
        "patience": 10
    },
    "Metrics": {
        "metrics_list": [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "confusion_matrix",
            "classification_report"
        ]
    },
    "Logger": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_handler": True,
        "console_handler": True,
        "log_file": "logs/ml_project.log"
    }
}

# Supported model names
SUPPORTED_MODELS: List[str] = [
    "LogisticRegression",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "SVM",
    "NaiveBayes",
    "KNN"
]

# Supported metrics
SUPPORTED_METRICS: List[str] = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "average_precision",
    "confusion_matrix",
    "classification_report"
]

# File extensions
MODEL_EXTENSIONS: List[str] = [".pkl", ".joblib", ".pth", ".h5"]
DATA_EXTENSIONS: List[str] = [".csv", ".parquet", ".json", ".xlsx"]
CONFIG_EXTENSIONS: List[str] = [".yaml", ".yml", ".json"]

# Data column mappings
TARGET_COLUMN: str = "Loan_Status"
ID_COLUMN: str = "Loan_ID"

# Feature engineering parameters
FEATURE_ENGINEERING: Dict[str, Any] = {
    "income_bins": [0, 3000, 6000, 10000, float('inf')],
    "income_labels": ['Low', 'Medium', 'High', 'VeryHigh'],
    "interest_rate": 0.08,  # 8% annual interest rate for EMI calculation
    "kmeans_clusters": 3,
    "polynomial_degree": 2
}

# Validation parameters
VALIDATION: Dict[str, Any] = {
    "min_samples": 10,
    "max_missing_percentage": 50.0,
    "outlier_threshold": 3.0,  # Z-score threshold
    "correlation_threshold": 0.95
}

# Output directories
OUTPUT_DIRS: List[str] = [
    "outputs",
    "outputs/models",
    "outputs/curves",
    "outputs/reports",
    "logs",
    "swanlog"
]

# Logging configuration
LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "logs/ml_project.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}

# Error messages
ERROR_MESSAGES: Dict[str, str] = {
    "MODEL_NOT_FITTED": "Model must be fitted before making predictions",
    "INVALID_CONFIG": "Invalid configuration provided",
    "FILE_NOT_FOUND": "File not found: {}",
    "INVALID_DATA": "Invalid data provided",
    "MISSING_TARGET": "Target column not found in data",
    "EMPTY_DATASET": "Dataset is empty",
    "INVALID_MODEL": "Unsupported model type: {}",
    "INVALID_METRIC": "Unsupported metric: {}",
    "CONFIG_VALIDATION_FAILED": "Configuration validation failed: {}"
}

# Success messages
SUCCESS_MESSAGES: Dict[str, str] = {
    "MODEL_SAVED": "Model saved successfully to: {}",
    "MODEL_LOADED": "Model loaded successfully from: {}",
    "DATA_LOADED": "Data loaded successfully: {} samples, {} features",
    "TRAINING_COMPLETED": "Training completed successfully",
    "EVALUATION_COMPLETED": "Evaluation completed successfully",
    "PREPROCESSING_COMPLETED": "Data preprocessing completed successfully"
}

# Performance thresholds
PERFORMANCE_THRESHOLDS: Dict[str, float] = {
    "excellent_accuracy": 0.90,
    "good_accuracy": 0.80,
    "acceptable_accuracy": 0.70,
    "excellent_f1": 0.85,
    "good_f1": 0.75,
    "acceptable_f1": 0.65
}
