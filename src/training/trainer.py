"""
Advanced model trainer with production-grade features.

This module provides a comprehensive training system with:
- Registry-based component management
- Caching for expensive operations
- Async task execution
- Real-time monitoring and metrics
- Advanced configuration management
- Performance profiling
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ..core.interfaces import BaseTrainer
from ..core.logger import Logger
from ..core.registry import register_component, get_component
from ..core.cache import cached, get_cache_manager
from ..core.async_tasks import submit_function, TaskPriority
from ..core.monitoring import profile_function, add_metric, get_monitoring_system
from ..core.advanced_config import AdvancedConfigManager
from ..core.exceptions import TrainingError, ModelError


@register_component("advanced_trainer", "trainers")
class AdvancedModelTrainer(BaseTrainer):
    """
    Production-grade model trainer with advanced features.
    
    Features:
    - Component registry integration
    - Intelligent caching
    - Async task execution
    - Real-time monitoring
    - Performance profiling
    - Advanced configuration management
    """
    
    def __init__(
        self, 
        config_manager: AdvancedConfigManager,
        logger: Optional[Logger] = None
    ):
        """
        Initialize the advanced trainer.
        
        Args:
            config_manager: Advanced configuration manager
            logger: Logger instance
        """
        super().__init__(None, config_manager.get_section("Train"))
        self.config_manager = config_manager
        self.logger = logger
        self.monitoring = get_monitoring_system()
        self.cache_manager = get_cache_manager()
        
        # Training configuration
        self.train_config = config_manager.get_section("Train")
        self.model_config = config_manager.get_section("Model")
        self.metrics_config = config_manager.get_section("Metrics")
        
        # Training parameters
        self.cv_folds = self.train_config.get('cv_folds', 5)
        self.random_state = self.train_config.get('random_seed', 42)
        self.verbose = self.train_config.get('verbose', 1)
        self.early_stopping_rounds = self.train_config.get('early_stopping_rounds', 10)
        self.save_best_model = self.train_config.get('save_best_model', True)
        
        # Cross-validation setup
        self.cv_strategy = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Training state
        self.training_history: List[Dict[str, Any]] = []
        self.best_model = None
        self.best_score = 0.0
        
        if self.logger:
            self.logger.info("AdvancedModelTrainer initialized with production features")
    
    @profile_function
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: Optional[np.ndarray] = None, 
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the model with advanced features.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training results dictionary
        """
        if self.model is None:
            # Try to get model from registry if not set
            model_name = self.model_config.get("model_name", "XGBoost")
            try:
                from ..core.registry import get_component
                self.model = get_component(model_name.lower(), "models", 
                                         config=self.model_config, logger=self.logger)
                if self.logger:
                    self.logger.info(f"Auto-loaded model: {model_name}")
            except Exception as e:
                raise TrainingError(f"No model set for training and failed to auto-load: {e}")
        
        start_time = time.time()
        
        try:
            if self.logger:
                self.logger.info(f"Starting advanced training for {self.model.model_name}")
            
            # Add training metrics
            add_metric("training.samples", len(X_train))
            add_metric("training.features", X_train.shape[1])
            add_metric("training.cv_folds", self.cv_folds)
            
            # Perform cross-validation with caching
            cv_results = self._perform_cross_validation(X_train, y_train)
            
            # Train final model
            final_model_results = self._train_final_model(X_train, y_train, X_val, y_val)
            
            # Combine results
            training_results = {
                'cv_results': cv_results,
                'final_model': final_model_results,
                'training_time': time.time() - start_time,
                'model_name': self.model.model_name,
                'config_hash': self.config_manager.get_config_hash()
            }
            
            # Store training history
            self.training_history.append(training_results)
            
            # Add performance metrics
            add_metric("training.cv_score", cv_results['cv_mean'])
            add_metric("training.cv_std", cv_results['cv_std'])
            add_metric("training.duration", training_results['training_time'])
            
            if self.logger:
                self.logger.info(f"Training completed. CV Score: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
                self.logger.info(f"Training time: {training_results['training_time']:.2f} seconds")
            
            return training_results
            
        except Exception as e:
            add_metric("training.errors", 1)
            error_msg = f"Training failed: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise TrainingError(error_msg) from e
    
    @cached(ttl=3600)  # Cache for 1 hour
    def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform cross-validation with caching.
        
        Args:
            X: Features
            y: Targets
            
        Returns:
            Cross-validation results
        """
        if self.logger:
            self.logger.info(f"Performing {self.cv_folds}-fold cross-validation...")
        
        # Generate cache key based on data and configuration
        model_name = getattr(self.model, 'model_name', 'Unknown')
        # The model object itself is the estimator
        model_obj = self.model
        cache_key = f"cv_{model_name}_{hash(X.tobytes())}_{self.cv_folds}_{self.random_state}"
        
        try:
            # Perform cross-validation manually to avoid scikit-learn cloning issues
            cv_scores = []
            for train_idx, val_idx in self.cv_strategy.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Create a new model instance for each fold
                fold_model = self.build_model()
                fold_model.fit(X_train_fold, y_train_fold)
                y_pred = fold_model.predict(X_val_fold)
                
                # Calculate F1 score
                from sklearn.metrics import f1_score
                score = f1_score(y_val_fold, y_pred, average='weighted')
                cv_scores.append(score)
            
            cv_scores = np.array(cv_scores)
            
            cv_results = {
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_min': cv_scores.min(),
                'cv_max': cv_scores.max()
            }
            
            # Add detailed metrics for each fold
            for i, score in enumerate(cv_scores):
                add_metric(f"training.cv_fold_{i}", score)
            
            return cv_results
            
        except Exception as e:
            error_msg = f"Cross-validation failed: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise TrainingError(error_msg) from e
    
    @profile_function
    def _train_final_model(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the final model with monitoring.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Final model training results
        """
        if self.logger:
            self.logger.info(f"Training final {self.model.model_name} model...")
        
        start_time = time.time()
        
        try:
            # Train the model
            self.model.fit(X_train, y_train, X_val, y_val)
            
            # Calculate training metrics
            train_predictions = self.model.predict(X_train)
            train_metrics = self._calculate_metrics(y_train, train_predictions, X_train)
            
            # Calculate validation metrics if validation data provided
            val_metrics = {}
            if X_val is not None and y_val is not None:
                val_predictions = self.model.predict(X_val)
                val_metrics = self._calculate_metrics(y_val, val_predictions, X_val)
                
                # Check if this is the best model
                val_score = val_metrics.get('f1_score', 0)
                if val_score > self.best_score:
                    self.best_score = val_score
                    self.best_model = self.model
                    
                    if self.save_best_model:
                        self._save_best_model()
            
            training_time = time.time() - start_time
            
            results = {
                'training_metrics': train_metrics,
                'validation_metrics': val_metrics,
                'training_time': training_time,
                'model_fitted': True
            }
            
            # Add metrics to monitoring
            for metric_name, value in train_metrics.items():
                add_metric(f"training.{metric_name}", value)
            
            for metric_name, value in val_metrics.items():
                add_metric(f"validation.{metric_name}", value)
            
            return results
            
        except Exception as e:
            error_msg = f"Final model training failed: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise TrainingError(error_msg) from e
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, X: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            X: Features (optional, needed for ROC-AUC calculation)
            
        Returns:
            Dictionary of metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted')
            }
            
            # Add ROC-AUC if binary classification and features are provided
            if len(np.unique(y_true)) == 2 and X is not None:
                y_proba = self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else None
                if y_proba is not None:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            
            return metrics
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error calculating metrics: {e}")
            return {}
    
    def _save_best_model(self) -> None:
        """Save the best model."""
        try:
            model_save_path = self.config_manager.get("Train.model_save_path", "outputs/models/best_model.pkl")
            self.best_model.save_model(model_save_path)
            
            if self.logger:
                self.logger.info(f"Best model saved to: {model_save_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save best model: {e}")
    
    async def train_async(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: Optional[np.ndarray] = None, 
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the model asynchronously.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Training results
        """
        if self.logger:
            self.logger.info("Starting async training...")
        
        # Submit training as async task
        task_id = submit_function(
            self.train,
            args=(X_train, y_train, X_val, y_val),
            priority=TaskPriority.HIGH
        )
        
        # Wait for completion
        from ..core.async_tasks import get_scheduler
        scheduler = get_scheduler()
        result = await scheduler.wait_for_task(task_id)
        
        if result.status.value == "completed":
            return result.result
        else:
            raise TrainingError(f"Async training failed: {result.error}")
    
    def hyperparameter_search(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        param_grid: Dict[str, List[Any]],
        cv_folds: int = 3
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter search with caching and monitoring.
        
        Args:
            X_train: Training features
            y_train: Training targets
            param_grid: Parameter grid for search
            cv_folds: Number of CV folds
            
        Returns:
            Hyperparameter search results
        """
        if self.logger:
            self.logger.info("Starting hyperparameter search...")
        
        from sklearn.model_selection import GridSearchCV
        
        try:
            # Create grid search with monitoring
            grid_search = GridSearchCV(
                self.model.model,
                param_grid,
                cv=cv_folds,
                scoring='f1',
                n_jobs=-1,
                verbose=self.verbose
            )
            
            # Perform search
            grid_search.fit(X_train, y_train)
            
            # Extract results
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_estimator': grid_search.best_estimator_,
                'cv_results': grid_search.cv_results_
            }
            
            # Update model with best parameters
            self.model.model = grid_search.best_estimator_
            
            # Add metrics
            add_metric("hyperparameter_search.best_score", grid_search.best_score_)
            add_metric("hyperparameter_search.n_combinations", len(grid_search.cv_results_['params']))
            
            if self.logger:
                self.logger.info(f"Hyperparameter search completed. Best score: {grid_search.best_score_:.4f}")
                self.logger.info(f"Best parameters: {grid_search.best_params_}")
            
            return results
            
        except Exception as e:
            error_msg = f"Hyperparameter search failed: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise TrainingError(error_msg) from e
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self.training_history.copy()
    
    def build_model(self):
        """Build a new model instance for cross-validation."""
        if hasattr(self.model, 'build_model'):
            return self.model.build_model()
        else:
            # Fallback: create a new instance of the same class
            model_class = self.model.__class__
            return model_class(self.model_config, self.logger)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.training_history:
            return {}
        
        latest_training = self.training_history[-1]
        cv_results = latest_training.get('cv_results', {})
        
        return {
            'best_cv_score': cv_results.get('cv_mean', 0),
            'cv_std': cv_results.get('cv_std', 0),
            'training_time': latest_training.get('training_time', 0),
            'model_name': latest_training.get('model_name', 'unknown'),
            'total_trainings': len(self.training_history)
        }
    
    def export_training_report(self, filepath: Union[str, Path]) -> None:
        """Export comprehensive training report."""
        import json
        
        report = {
            'training_history': self.training_history,
            'performance_summary': self.get_performance_summary(),
            'configuration': {
                'train_config': self.train_config,
                'model_config': self.model_config,
                'config_hash': self.config_manager.get_config_hash()
            },
            'monitoring_metrics': self.monitoring.get_metric_summary("training.cv_score"),
            'performance_profiles': self.monitoring.get_performance_profiles()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        if self.logger:
            self.logger.info(f"Training report exported to: {filepath}")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on given data.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.model or not hasattr(self.model, 'predict'):
            raise TrainingError("Model not trained or invalid")
        
        try:
            # Make predictions
            y_pred = self.model.predict(X)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y, y_pred, X)
            
            if self.logger:
                self.logger.info(f"Model evaluation completed. Metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            error_msg = f"Model evaluation failed: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise TrainingError(error_msg) from e
