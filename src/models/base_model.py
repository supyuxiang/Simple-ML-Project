"""
Base model class for ML project.

This module provides the foundational classes and interfaces for all machine learning
models in the project, including comprehensive error handling, validation, and logging.
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

from ..core.interfaces import BaseModel
from ..core.constants import (
    DEFAULT_CONFIG,
    ERROR_MESSAGES,
    SUCCESS_MESSAGES,
    TARGET_COLUMN
)
from ..core.exceptions import (
    ModelError,
    ModelNotFittedError,
    TrainingError,
    ValidationError
)
from ..core.logger import Logger
from ..core.validators import ModelValidator


class BaseMLModel(BaseModel):
    """
    Base class for all machine learning models in the ML project.
    
    This class extends the core BaseModel with ML-specific functionality including:
    - Comprehensive model state management
    - Feature importance calculation
    - Cross-validation support
    - Model persistence (save/load)
    - Performance metrics tracking
    - Validation and error handling
    
    Attributes:
        config (Dict[str, Any]): Model configuration parameters
        logger (Optional[Logger]): Logger instance for tracking operations
        model_name (str): Name of the model
        model_type (str): Type of model (classification/regression)
        model_params (Dict[str, Any]): Model hyperparameters
        model (Optional[BaseEstimator]): The actual model instance
        is_fitted (bool): Whether the model has been fitted
        training_history (Dict[str, Any]): Training history and metrics
        feature_importance (Optional[np.ndarray]): Feature importance scores
        classes_ (Optional[np.ndarray]): Class labels for classification
        n_features_in_ (Optional[int]): Number of features seen during fit
        train_metrics (Dict[str, float]): Training set metrics
        val_metrics (Dict[str, float]): Validation set metrics
        test_metrics (Dict[str, float]): Test set metrics
        validator (ModelValidator): Model validation utility
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None) -> None:
        """
        Initialize the base ML model with configuration and logger.
        
        Args:
            config: Model configuration dictionary containing:
                - model_name: Name of the model
                - model_type: Type of model (classification/regression)
                - model_params: Model hyperparameters
            logger: Optional logger instance for tracking operations
            
        Raises:
            ModelError: If configuration is invalid
        """
        super().__init__(config)
        self.logger = logger
        self.validator = ModelValidator()
        
        # Model configuration
        self.model_name: str = config.get('model_name', 'Unknown')
        self.model_type: str = config.get('model_type', DEFAULT_CONFIG['Model']['model_type'])
        self.model_params: Dict[str, Any] = config.get('model_params', {})
        
        # Model state
        self.training_history: Dict[str, Any] = {}
        self.feature_importance: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        
        # Performance metrics
        self.train_metrics: Dict[str, float] = {}
        self.val_metrics: Dict[str, float] = {}
        self.test_metrics: Dict[str, float] = {}
        
        # Validate model configuration
        self._validate_model_config()
        
        if self.logger:
            self.logger.info(f"Initialized {self.model_name} model (type: {self.model_type})")
            self.logger.debug(f"Model parameters: {self.model_params}")
    
    def _validate_model_config(self) -> None:
        """
        Validate model configuration parameters.
        
        Raises:
            ModelError: If configuration is invalid
        """
        if not self.model_name or self.model_name == 'Unknown':
            raise ModelError("Model name must be specified in configuration")
        
        if self.model_type not in ['classification', 'regression']:
            raise ModelError(f"Invalid model type: {self.model_type}. Must be 'classification' or 'regression'")
        
        if not isinstance(self.model_params, dict):
            raise ModelError("Model parameters must be a dictionary")
    
    @abstractmethod
    def build_model(self) -> Any:
        """
        Build the model architecture
        
        Returns:
            The built model
        """
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> 'BaseMLModel':
        """
        Train the model
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Self for method chaining
        """
        if self.logger:
            self.logger.info(f"Training {self.model_name} model...")
        
        # Build model if not already built
        if self.model is None:
            self.model = self.build_model()
        
        # Store feature information
        self.n_features_in_ = X.shape[1]
        if hasattr(y, 'unique'):
            self.classes_ = np.unique(y)
        
        # Train the model
        self._fit_model(X, y, X_val, y_val)
        
        # Calculate feature importance if available
        self._calculate_feature_importance()
        
        self.is_fitted = True
        
        if self.logger:
            self.logger.info(f"{self.model_name} model training completed")
        
        return self
    
    @abstractmethod
    def _fit_model(self, X: np.ndarray, y: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """
        Internal method to fit the model
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with comprehensive validation and error handling.
        
        Args:
            X: Feature matrix to make predictions on
            
        Returns:
            Array of predictions
            
        Raises:
            ModelNotFittedError: If model hasn't been fitted
            ValidationError: If input validation fails
        """
        if not self.is_fitted:
            raise ModelNotFittedError(self.model_name)
        
        # Validate input
        if X is None or X.size == 0:
            raise ValidationError("Input features cannot be None or empty")
        
        if not isinstance(X, np.ndarray):
            raise ValidationError(f"Input must be numpy array, got {type(X)}")
        
        if X.ndim != 2:
            raise ValidationError(f"Input must be 2D array, got {X.ndim}D")
        
        # Check feature count consistency
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValidationError(
                f"Feature count mismatch: expected {self.n_features_in_}, got {X.shape[1]}"
            )
        
        try:
            if self.logger:
                self.logger.debug(f"Making predictions with {self.model_name} model on {X.shape[0]} samples")
            
            predictions = self._predict(X)
            
            # Validate predictions
            prediction_validation = self.validator.validate_predictions(
                np.zeros(X.shape[0]), predictions  # Dummy y_true for validation
            )
            if not prediction_validation['is_valid']:
                raise ValidationError(f"Prediction validation failed: {prediction_validation['errors']}")
            
            if self.logger:
                self.logger.debug(f"Predictions completed successfully. Shape: {predictions.shape}")
            
            return predictions
            
        except Exception as e:
            error_msg = f"Prediction failed for {self.model_name}: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise ModelError(error_msg)
    
    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Internal method to make predictions
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features to predict
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.logger:
            self.logger.debug(f"Making probability predictions with {self.model_name} model")
        
        probabilities = self._predict_proba(X)
        
        if self.logger:
            self.logger.debug(f"Probability predictions completed. Shape: {probabilities.shape}")
        
        return probabilities
    
    @abstractmethod
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Internal method to predict probabilities
        
        Args:
            X: Features to predict
            
        Returns:
            Class probabilities
        """
        pass
    
    def _calculate_feature_importance(self) -> None:
        """
        Calculate feature importance if the model supports it
        """
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            self.feature_importance = np.abs(self.model.coef_[0])
        else:
            self.feature_importance = None
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance scores
        """
        if self.feature_importance is None:
            return None
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance))]
        
        importance_dict = dict(zip(feature_names, self.feature_importance))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'is_fitted': self.is_fitted,
            'n_features_in_': self.n_features_in_,
            'classes_': self.classes_.tolist() if self.classes_ is not None else None,
            'has_feature_importance': self.feature_importance is not None
        }
        
        if self.feature_importance is not None:
            info['feature_importance_available'] = True
            info['top_features'] = self.get_feature_importance()
        
        return info
    
    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save the trained model with comprehensive metadata.
        
        Args:
            path: Path to save the model (supports .pkl, .joblib extensions)
            
        Raises:
            ModelNotFittedError: If model hasn't been fitted
            ModelError: If saving fails
        """
        if not self.is_fitted:
            raise ModelNotFittedError(self.model_name)
        
        save_path = Path(path)
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate file extension
        if save_path.suffix.lower() not in ['.pkl', '.joblib']:
            save_path = save_path.with_suffix('.pkl')
        
        try:
            if self.logger:
                self.logger.info(f"Saving {self.model_name} model to {save_path}")
            
            # Prepare model data with comprehensive metadata
            model_data = {
                'model': self.model,
                'model_name': self.model_name,
                'model_type': self.model_type,
                'model_params': self.model_params,
                'classes_': self.classes_,
                'n_features_in_': self.n_features_in_,
                'feature_importance': self.feature_importance,
                'training_history': self.training_history,
                'train_metrics': self.train_metrics,
                'val_metrics': self.val_metrics,
                'test_metrics': self.test_metrics,
                'is_fitted': self.is_fitted,
                'feature_names': getattr(self, 'feature_names', None),
                'target_name': getattr(self, 'target_name', TARGET_COLUMN)
            }
            
            # Save using joblib (preferred for sklearn models)
            joblib.dump(model_data, save_path, compress=3)
            
            if self.logger:
                self.logger.info(SUCCESS_MESSAGES['MODEL_SAVED'].format(save_path))
                
        except Exception as e:
            error_msg = f"Failed to save model to {save_path}: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise ModelError(error_msg)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.
        
        Args:
            deep: If True, will return the parameters for this estimator and
                  contained subobjects that are estimators.
                  
        Returns:
            Parameter names mapped to their values.
        """
        # Get default parameters from the model if it exists
        if hasattr(self, 'build_model'):
            try:
                # Create a temporary model to get default parameters
                temp_model = self.build_model()
                if hasattr(temp_model, 'get_params'):
                    default_params = temp_model.get_params()
                    # Merge with our model_params
                    params = default_params.copy()
                    params.update(self.model_params)
                    return params
            except:
                pass
        
        return self.model_params.copy()
    
    def set_params(self, **params) -> 'BaseMLModel':
        """
        Set the parameters of this estimator.
        
        Args:
            **params: Estimator parameters.
            
        Returns:
            Self.
        """
        # Update model parameters
        for key, value in params.items():
            self.model_params[key] = value
        
        # Rebuild model with new parameters
        if hasattr(self, 'build_model'):
            self.model = self.build_model()
        
        return self
    
    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load a trained model with comprehensive validation.
        
        Args:
            path: Path to load the model from
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ModelError: If loading or validation fails
        """
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        try:
            if self.logger:
                self.logger.info(f"Loading {self.model_name} model from {load_path}")
            
            # Load model data
            model_data = joblib.load(load_path)
            
            # Validate loaded data structure
            required_keys = [
                'model', 'model_name', 'model_type', 'model_params',
                'is_fitted'
            ]
            for key in required_keys:
                if key not in model_data:
                    raise ModelError(f"Missing required key in model file: {key}")
            
            # Restore model state
            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.model_type = model_data['model_type']
            self.model_params = model_data['model_params']
            self.classes_ = model_data.get('classes_')
            self.n_features_in_ = model_data.get('n_features_in_')
            self.feature_importance = model_data.get('feature_importance')
            self.training_history = model_data.get('training_history', {})
            self.train_metrics = model_data.get('train_metrics', {})
            self.val_metrics = model_data.get('val_metrics', {})
            self.test_metrics = model_data.get('test_metrics', {})
            self.is_fitted = model_data['is_fitted']
            
            # Restore additional attributes if available
            if 'feature_names' in model_data:
                self.feature_names = model_data['feature_names']
            if 'target_name' in model_data:
                self.target_name = model_data['target_name']
            
            # Validate loaded model
            if not self.is_fitted:
                raise ModelError("Loaded model is not fitted")
            
            if self.logger:
                self.logger.info(SUCCESS_MESSAGES['MODEL_LOADED'].format(load_path))
                self.logger.debug(f"Model info: {self.get_model_info()}")
                
        except Exception as e:
            error_msg = f"Failed to load model from {load_path}: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise ModelError(error_msg)
    
    def update_metrics(self, metrics: Dict[str, float], split: str = 'train') -> None:
        """
        Update model metrics
        
        Args:
            metrics: Dictionary of metrics
            split: Data split ('train', 'val', 'test')
        """
        if split == 'train':
            self.train_metrics.update(metrics)
        elif split == 'val':
            self.val_metrics.update(metrics)
        elif split == 'test':
            self.test_metrics.update(metrics)
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary of all metrics
        
        Returns:
            Dictionary containing metrics for all splits
        """
        return {
            'train': self.train_metrics,
            'validation': self.val_metrics,
            'test': self.test_metrics
        }
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary containing cross-validation results
        """
        from sklearn.model_selection import cross_val_score
        
        if not self.is_fitted:
            # Build and fit model for cross-validation
            self.model = self.build_model()
            self._fit_model(X, y)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        cv_results = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        if self.logger:
            self.logger.info(f"Cross-validation results: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
        
        return cv_results
    
    def __str__(self) -> str:
        """
        String representation of the model
        """
        return f"{self.model_name}({self.model_type})"
    
    def __repr__(self) -> str:
        """
        Detailed string representation of the model
        """
        return f"{self.__class__.__name__}(model_name='{self.model_name}', model_type='{self.model_type}', is_fitted={self.is_fitted})"
