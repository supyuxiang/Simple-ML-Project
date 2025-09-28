"""
Validation utilities for the ML project.
This module provides validation functions for data, configuration, and models.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import yaml
import json

from .exceptions import (
    ConfigurationValidationError,
    InvalidDataError,
    EmptyDatasetError,
    MissingTargetError,
    DataQualityError,
    UnsupportedModelError,
    UnsupportedMetricError
)
from .constants import (
    SUPPORTED_MODELS,
    SUPPORTED_METRICS,
    TARGET_COLUMN,
    ID_COLUMN,
    VALIDATION,
    DEFAULT_CONFIG
)


class ConfigValidator:
    """Validator for configuration files and parameters."""
    
    @staticmethod
    def validate_config_file(config_path: str) -> Dict[str, Any]:
        """
        Validate and load configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ConfigurationValidationError: If configuration is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationValidationError(f"Configuration file not found: {config_path}")
        
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                raise ConfigurationValidationError(f"Unsupported configuration file format: {config_path.suffix}")
        except Exception as e:
            raise ConfigurationValidationError(f"Failed to load configuration file: {e}")
        
        return ConfigValidator.validate_config_dict(config)
    
    @staticmethod
    def validate_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ConfigurationValidationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ConfigurationValidationError("Configuration must be a dictionary")
        
        # Validate required sections
        required_sections = ['Model', 'Data', 'Train', 'Metrics']
        for section in required_sections:
            if section not in config:
                raise ConfigurationValidationError(f"Missing required configuration section: {section}")
        
        # Validate Model section
        ConfigValidator._validate_model_config(config['Model'])
        
        # Validate Data section
        ConfigValidator._validate_data_config(config['Data'])
        
        # Validate Train section
        ConfigValidator._validate_train_config(config['Train'])
        
        # Validate Metrics section
        ConfigValidator._validate_metrics_config(config['Metrics'])
        
        return config
    
    @staticmethod
    def _validate_model_config(model_config: Dict[str, Any]) -> None:
        """Validate model configuration section."""
        if 'model_name' not in model_config:
            raise ConfigurationValidationError("Model configuration missing 'model_name'")
        
        model_name = model_config['model_name']
        if model_name not in SUPPORTED_MODELS:
            raise UnsupportedModelError(model_name)
        
        if 'model_type' not in model_config:
            raise ConfigurationValidationError("Model configuration missing 'model_type'")
        
        if 'model_params' not in model_config:
            raise ConfigurationValidationError("Model configuration missing 'model_params'")
        
        if not isinstance(model_config['model_params'], dict):
            raise ConfigurationValidationError("Model parameters must be a dictionary")
    
    @staticmethod
    def _validate_data_config(data_config: Dict[str, Any]) -> None:
        """Validate data configuration section."""
        required_params = ['test_size', 'random_seed']
        for param in required_params:
            if param not in data_config:
                raise ConfigurationValidationError(f"Data configuration missing '{param}'")
        
        # Validate test_size
        test_size = data_config['test_size']
        if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
            raise ConfigurationValidationError("test_size must be a number between 0 and 1")
        
        # Validate random_seed
        random_seed = data_config['random_seed']
        if not isinstance(random_seed, int) or random_seed < 0:
            raise ConfigurationValidationError("random_seed must be a non-negative integer")
    
    @staticmethod
    def _validate_train_config(train_config: Dict[str, Any]) -> None:
        """Validate training configuration section."""
        if 'cv_folds' not in train_config:
            raise ConfigurationValidationError("Train configuration missing 'cv_folds'")
        
        cv_folds = train_config['cv_folds']
        if not isinstance(cv_folds, int) or cv_folds < 2:
            raise ConfigurationValidationError("cv_folds must be an integer >= 2")
    
    @staticmethod
    def _validate_metrics_config(metrics_config: Dict[str, Any]) -> None:
        """Validate metrics configuration section."""
        if 'metrics_list' not in metrics_config:
            raise ConfigurationValidationError("Metrics configuration missing 'metrics_list'")
        
        metrics_list = metrics_config['metrics_list']
        if not isinstance(metrics_list, list):
            raise ConfigurationValidationError("metrics_list must be a list")
        
        for metric in metrics_list:
            if metric not in SUPPORTED_METRICS:
                raise UnsupportedMetricError(metric)


class DataValidator:
    """Validator for data quality and structure."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config or VALIDATION
    
    def validate_dataframe(self, df: pd.DataFrame, 
                          require_target: bool = True,
                          target_name: str = TARGET_COLUMN) -> Dict[str, Any]:
        """
        Validate DataFrame structure and quality.
        
        Args:
            df: DataFrame to validate
            require_target: Whether target column is required
            target_name: Name of target column
            
        Returns:
            Validation results dictionary
            
        Raises:
            EmptyDatasetError: If dataset is empty
            MissingTargetError: If target column is missing
            DataQualityError: If data quality is below threshold
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 1.0,
            'shape': df.shape,
            'missing_percentage': {},
            'data_types': df.dtypes.to_dict()
        }
        
        # Check if dataset is empty
        if df.empty:
            raise EmptyDatasetError()
        
        # Check minimum samples
        min_samples = self.config.get('min_samples', 10)
        if len(df) < min_samples:
            validation_results['errors'].append(f"Dataset has only {len(df)} samples, minimum required: {min_samples}")
            validation_results['is_valid'] = False
        
        # Check for target column if required
        if require_target and target_name not in df.columns:
            raise MissingTargetError(target_name)
        
        # Check missing values
        missing_percentage = (df.isnull().sum() / len(df) * 100).to_dict()
        validation_results['missing_percentage'] = missing_percentage
        
        max_missing = self.config.get('max_missing_percentage', 50.0)
        high_missing_cols = [col for col, pct in missing_percentage.items() if pct > max_missing]
        
        if high_missing_cols:
            validation_results['warnings'].append(f"Columns with high missing values: {high_missing_cols}")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results['warnings'].append(f"Found {duplicate_count} duplicate rows")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(df, missing_percentage, duplicate_count)
        validation_results['quality_score'] = quality_score
        
        # Check quality threshold
        quality_threshold = self.config.get('quality_threshold', 0.5)
        if quality_score < quality_threshold:
            validation_results['errors'].append(f"Data quality score {quality_score:.2f} below threshold {quality_threshold:.2f}")
            validation_results['is_valid'] = False
        
        return validation_results
    
    def validate_features(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate feature matrix.
        
        Args:
            X: Feature matrix
            feature_names: Optional feature names
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'shape': X.shape,
            'has_nan': False,
            'has_inf': False,
            'feature_names': feature_names
        }
        
        # Check for NaN values
        if np.isnan(X).any():
            validation_results['has_nan'] = True
            validation_results['errors'].append("Feature matrix contains NaN values")
            validation_results['is_valid'] = False
        
        # Check for infinite values
        if np.isinf(X).any():
            validation_results['has_inf'] = True
            validation_results['errors'].append("Feature matrix contains infinite values")
            validation_results['is_valid'] = False
        
        # Check for constant features
        constant_features = []
        for i in range(X.shape[1]):
            if np.std(X[:, i]) == 0:
                constant_features.append(i)
        
        if constant_features:
            validation_results['warnings'].append(f"Found {len(constant_features)} constant features")
        
        # Check for highly correlated features
        if X.shape[1] > 1:
            correlation_threshold = self.config.get('correlation_threshold', 0.95)
            corr_matrix = np.corrcoef(X.T)
            high_corr_pairs = []
            
            for i in range(corr_matrix.shape[0]):
                for j in range(i + 1, corr_matrix.shape[1]):
                    if abs(corr_matrix[i, j]) > correlation_threshold:
                        high_corr_pairs.append((i, j, corr_matrix[i, j]))
            
            if high_corr_pairs:
                validation_results['warnings'].append(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
        
        return validation_results
    
    def validate_target(self, y: np.ndarray, task_type: str = 'classification') -> Dict[str, Any]:
        """
        Validate target variable.
        
        Args:
            y: Target variable
            task_type: Type of task ('classification' or 'regression')
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'shape': y.shape,
            'unique_values': len(np.unique(y)),
            'has_nan': False,
            'has_inf': False
        }
        
        # Check for NaN values
        if np.isnan(y).any():
            validation_results['has_nan'] = True
            validation_results['errors'].append("Target variable contains NaN values")
            validation_results['is_valid'] = False
        
        # Check for infinite values
        if np.isinf(y).any():
            validation_results['has_inf'] = True
            validation_results['errors'].append("Target variable contains infinite values")
            validation_results['is_valid'] = False
        
        if task_type == 'classification':
            # Check for class balance
            unique_values, counts = np.unique(y, return_counts=True)
            min_count = np.min(counts)
            max_count = np.max(counts)
            
            if min_count < 2:
                validation_results['warnings'].append("Some classes have very few samples")
            
            # Check for class imbalance
            imbalance_ratio = max_count / min_count
            if imbalance_ratio > 10:
                validation_results['warnings'].append(f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f})")
        
        return validation_results
    
    def _calculate_quality_score(self, df: pd.DataFrame, 
                                missing_percentage: Dict[str, float], 
                                duplicate_count: int) -> float:
        """
        Calculate data quality score.
        
        Args:
            df: DataFrame
            missing_percentage: Missing value percentages
            duplicate_count: Number of duplicate rows
            
        Returns:
            Quality score between 0 and 1
        """
        # Base score
        score = 1.0
        
        # Penalize high missing values
        avg_missing = np.mean(list(missing_percentage.values()))
        score -= min(avg_missing / 100, 0.5)  # Max penalty of 0.5
        
        # Penalize duplicates
        duplicate_penalty = min(duplicate_count / len(df), 0.3)  # Max penalty of 0.3
        score -= duplicate_penalty
        
        # Penalize constant columns
        constant_cols = df.nunique() == 1
        constant_penalty = constant_cols.sum() / len(df.columns) * 0.2  # Max penalty of 0.2
        score -= constant_penalty
        
        return max(score, 0.0)


class ModelValidator:
    """Validator for model parameters and predictions."""
    
    @staticmethod
    def validate_model_name(model_name: str) -> None:
        """
        Validate model name.
        
        Args:
            model_name: Name of the model
            
        Raises:
            UnsupportedModelError: If model is not supported
        """
        if model_name not in SUPPORTED_MODELS:
            raise UnsupportedModelError(model_name)
    
    @staticmethod
    def validate_metrics(metrics: List[str]) -> None:
        """
        Validate metrics list.
        
        Args:
            metrics: List of metric names
            
        Raises:
            UnsupportedMetricError: If metric is not supported
        """
        for metric in metrics:
            if metric not in SUPPORTED_METRICS:
                raise UnsupportedMetricError(metric)
    
    @staticmethod
    def validate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Validate predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check shapes match
        if y_true.shape != y_pred.shape:
            validation_results['errors'].append(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
            validation_results['is_valid'] = False
        
        # Check for NaN in predictions
        if np.isnan(y_pred).any():
            validation_results['errors'].append("Predictions contain NaN values")
            validation_results['is_valid'] = False
        
        # Check for infinite values in predictions
        if np.isinf(y_pred).any():
            validation_results['errors'].append("Predictions contain infinite values")
            validation_results['is_valid'] = False
        
        return validation_results
