"""
Model trainer for ML1 project
Handles model training with advanced features
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ..core.base import BaseTrainer
from ..core.logger import Logger


class ModelTrainer(BaseTrainer):
    """
    Advanced model trainer with cross-validation and early stopping
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        """
        Initialize the trainer
        
        Args:
            config: Training configuration
            logger: Logger instance
        """
        super().__init__(None, config)  # model will be set later
        self.logger = logger
        
        # Training parameters
        self.cv_folds = config.get('cv_folds', 5)
        self.random_state = config.get('random_seed', 42)
        self.verbose = config.get('verbose', 1)
        
        # Cross-validation setup
        self.cv_strategy = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        if self.logger:
            self.logger.info("ModelTrainer initialized")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model with cross-validation
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model must be set before training")
        
        if self.logger:
            self.logger.info(f"Training {self.model.model_name} model...")
        
        # Perform cross-validation
        cv_scores = self._cross_validate(X_train, y_train)
        
        # Train on full training set
        self.model.fit(X_train, y_train, X_val, y_val)
        
        # Evaluate on validation set if provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val)
            val_probabilities = self.model.predict_proba(X_val)
            val_metrics = self._compute_metrics(y_val, val_predictions, val_probabilities)
        
        # Store training history
        self.training_history = {
            'cv_scores': cv_scores,
            'val_metrics': val_metrics,
            'n_samples': X_train.shape[0],
            'n_features': X_train.shape[1]
        }
        
        if self.logger:
            self.logger.info(f"Training completed. CV Score: {cv_scores['mean']:.4f} ± {cv_scores['std']:.4f}")
        
        return self.training_history
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Cross-validation results
        """
        if self.logger:
            self.logger.info(f"Performing {self.cv_folds}-fold cross-validation...")
        
        # Build model if not already built
        if self.model.model is None:
            self.model.model = self.model.build_model()
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model.model,  # Use the underlying sklearn model
            X, y,
            cv=self.cv_strategy,
            scoring='accuracy',
            n_jobs=-1
        )
        
        cv_results = {
            'scores': cv_scores.tolist(),
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'min': cv_scores.min(),
            'max': cv_scores.max()
        }
        
        return cv_results
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Evaluation metrics
        """
        if not self.model.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Compute metrics
        metrics = self._compute_metrics(y, predictions, probabilities)
        
        if self.logger:
            self.logger.info(f"Evaluation completed. Accuracy: {metrics.get('accuracy', 0):.4f}")
        
        return metrics
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Add ROC AUC if probabilities are available
        if y_proba is not None and y_proba.shape[1] > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except ValueError:
                # Handle case where only one class is present
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def set_model(self, model) -> None:
        """
        Set the model to train
        
        Args:
            model: Model instance
        """
        self.model = model
        if self.logger:
            self.logger.info(f"Model set to {model.model_name}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get training summary
        
        Returns:
            Training summary dictionary
        """
        summary = {
            'model_name': self.model.model_name if self.model else 'Unknown',
            'training_history': self.training_history,
            'cv_folds': self.cv_folds,
            'random_state': self.random_state
        }
        
        return summary
