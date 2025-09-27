"""
Base model class for ML1 project
Provides common interface for all machine learning models
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import joblib
import pickle
from pathlib import Path

from ..core.base import BaseModel
from ..core.logger import Logger


class BaseMLModel(BaseModel):
    """
    Base class for all machine learning models in ML1 project
    Extends the core BaseModel with ML-specific functionality
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        """
        Initialize the base ML model
        
        Args:
            config: Model configuration dictionary
            logger: Logger instance
        """
        super().__init__(config)
        self.logger = logger
        self.model_name = config.get('model_name', 'Unknown')
        self.model_type = config.get('model_type', 'classification')
        self.model_params = config.get('model_params', {})
        
        # Model state
        self.training_history = {}
        self.feature_importance = None
        self.classes_ = None
        self.n_features_in_ = None
        
        # Performance metrics
        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}
        
        if self.logger:
            self.logger.info(f"Initialized {self.model_name} model")
    
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
        Make predictions
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.logger:
            self.logger.debug(f"Making predictions with {self.model_name} model")
        
        predictions = self._predict(X)
        
        if self.logger:
            self.logger.debug(f"Predictions completed. Shape: {predictions.shape}")
        
        return predictions
    
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
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model using joblib (preferred for sklearn models)
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
            'test_metrics': self.test_metrics
        }
        
        joblib.dump(model_data, save_path)
        
        if self.logger:
            self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model
        
        Args:
            path: Path to load the model from
        """
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        # Load model data
        model_data = joblib.load(load_path)
        
        # Restore model state
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.model_type = model_data['model_type']
        self.model_params = model_data['model_params']
        self.classes_ = model_data['classes_']
        self.n_features_in_ = model_data['n_features_in_']
        self.feature_importance = model_data['feature_importance']
        self.training_history = model_data['training_history']
        self.train_metrics = model_data['train_metrics']
        self.val_metrics = model_data['val_metrics']
        self.test_metrics = model_data['test_metrics']
        
        self.is_fitted = True
        
        if self.logger:
            self.logger.info(f"Model loaded from {load_path}")
    
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
