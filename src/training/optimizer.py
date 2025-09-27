"""
Hyperparameter optimizer for ML1 project
Contains hyperparameter optimization classes and methods
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from ..core.logger import Logger


class HyperparameterOptimizer:
    """
    Hyperparameter optimization for ML1 models
    Supports grid search, random search, and custom optimization strategies
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        """
        Initialize the hyperparameter optimizer
        
        Args:
            config: Optimization configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Optimization parameters
        self.optimization_method = config.get('optimization_method', 'grid_search')
        self.cv_folds = config.get('cv_folds', 5)
        self.n_iter = config.get('n_iter', 100)  # For random search
        self.scoring = config.get('scoring', 'accuracy')
        self.n_jobs = config.get('n_jobs', -1)
        self.random_state = config.get('random_state', 42)
        
        # Results storage
        self.optimization_results = {}
        self.best_params = {}
        self.best_score = {}
        
        if self.logger:
            self.logger.info("HyperparameterOptimizer initialized")
    
    def optimize_model(self, model, X: np.ndarray, y: np.ndarray, 
                      param_grid: Dict[str, List]) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a given model
        
        Args:
            model: Model to optimize
            X: Training features
            y: Training labels
            param_grid: Parameter grid for optimization
            
        Returns:
            Optimization results
        """
        if self.logger:
            self.logger.info(f"Starting hyperparameter optimization for {model.__class__.__name__}")
        
        # Create scorer
        scorer = make_scorer(accuracy_score)
        
        if self.optimization_method == 'grid_search':
            optimizer = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=self.cv_folds,
                scoring=scorer,
                n_jobs=self.n_jobs,
                verbose=1
            )
        elif self.optimization_method == 'random_search':
            optimizer = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=self.n_iter,
                cv=self.cv_folds,
                scoring=scorer,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1
            )
        else:
            raise ValueError(f"Unsupported optimization method: {self.optimization_method}")
        
        # Perform optimization
        optimizer.fit(X, y)
        
        # Store results
        model_name = model.__class__.__name__
        self.optimization_results[model_name] = {
            'best_params': optimizer.best_params_,
            'best_score': optimizer.best_score_,
            'best_estimator': optimizer.best_estimator_,
            'cv_results': optimizer.cv_results_
        }
        
        self.best_params[model_name] = optimizer.best_params_
        self.best_score[model_name] = optimizer.best_score_
        
        if self.logger:
            self.logger.info(f"Optimization completed for {model_name}")
            self.logger.info(f"Best score: {optimizer.best_score_:.4f}")
            self.logger.info(f"Best parameters: {optimizer.best_params_}")
        
        return self.optimization_results[model_name]
    
    def get_best_params(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get best parameters for a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Best parameters dictionary
        """
        return self.best_params.get(model_name)
    
    def get_best_score(self, model_name: str) -> Optional[float]:
        """
        Get best score for a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Best score
        """
        return self.best_score.get(model_name)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get summary of all optimization results
        
        Returns:
            Summary dictionary
        """
        summary = {
            'optimization_method': self.optimization_method,
            'cv_folds': self.cv_folds,
            'scoring': self.scoring,
            'models_optimized': list(self.optimization_results.keys()),
            'best_scores': self.best_score,
            'best_params': self.best_params
        }
        
        return summary
    
    def create_param_grid(self, model_name: str) -> Dict[str, List]:
        """
        Create parameter grid for different models
        
        Args:
            model_name: Name of the model
            
        Returns:
            Parameter grid dictionary
        """
        param_grids = {
            'LogisticRegression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000, 2000]
            },
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'SVC': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'probability': [True]
            },
            'XGBClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'LGBMClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'KNeighborsClassifier': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'GaussianNB': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            }
        }
        
        return param_grids.get(model_name, {})
    
    def optimize_multiple_models(self, models: List[Any], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize multiple models
        
        Args:
            models: List of models to optimize
            X: Training features
            y: Training labels
            
        Returns:
            Dictionary of optimization results
        """
        if self.logger:
            self.logger.info(f"Starting optimization for {len(models)} models")
        
        all_results = {}
        
        for model in models:
            model_name = model.__class__.__name__
            param_grid = self.create_param_grid(model_name)
            
            if param_grid:
                try:
                    result = self.optimize_model(model, X, y, param_grid)
                    all_results[model_name] = result
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error optimizing {model_name}: {e}")
                    all_results[model_name] = {'error': str(e)}
            else:
                if self.logger:
                    self.logger.warning(f"No parameter grid defined for {model_name}")
                all_results[model_name] = {'error': 'No parameter grid defined'}
        
        if self.logger:
            self.logger.info("Multi-model optimization completed")
        
        return all_results
    
    def get_best_model(self) -> Tuple[Optional[str], Optional[float]]:
        """
        Get the best performing model and its score
        
        Returns:
            Tuple of (model_name, best_score)
        """
        if not self.best_score:
            return None, None
        
        best_model = max(self.best_score.items(), key=lambda x: x[1])
        return best_model
    
    def plot_optimization_results(self, model_name: str, save_path: Optional[str] = None) -> None:
        """
        Plot optimization results for a model
        
        Args:
            model_name: Name of the model
            save_path: Path to save the plot
        """
        if model_name not in self.optimization_results:
            if self.logger:
                self.logger.warning(f"No optimization results found for {model_name}")
            return
        
        import matplotlib.pyplot as plt
        import pandas as pd
        
        results = self.optimization_results[model_name]
        cv_results = results['cv_results']
        
        # Create DataFrame from cv_results
        df_results = pd.DataFrame(cv_results)
        
        # Plot mean test scores
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Mean test scores
        plt.subplot(1, 2, 1)
        plt.plot(df_results['mean_test_score'])
        plt.title(f'{model_name} - Mean Test Scores')
        plt.xlabel('Parameter Combination')
        plt.ylabel('Mean Test Score')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Score distribution
        plt.subplot(1, 2, 2)
        plt.hist(df_results['mean_test_score'], bins=20, alpha=0.7)
        plt.title(f'{model_name} - Score Distribution')
        plt.xlabel('Mean Test Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            from pathlib import Path
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"Optimization results plot saved to {save_path}")
        
        plt.show()
    
    def save_optimization_results(self, file_path: str) -> None:
        """
        Save optimization results to file
        
        Args:
            file_path: Path to save the results
        """
        import joblib
        
        results_to_save = {
            'optimization_results': self.optimization_results,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'config': self.config
        }
        
        from pathlib import Path
        save_path = Path(file_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(results_to_save, save_path)
        
        if self.logger:
            self.logger.info(f"Optimization results saved to {save_path}")
    
    def load_optimization_results(self, file_path: str) -> None:
        """
        Load optimization results from file
        
        Args:
            file_path: Path to load the results from
        """
        import joblib
        
        from pathlib import Path
        load_path = Path(file_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Results file not found: {load_path}")
        
        results = joblib.load(load_path)
        
        self.optimization_results = results['optimization_results']
        self.best_params = results['best_params']
        self.best_score = results['best_score']
        
        if self.logger:
            self.logger.info(f"Optimization results loaded from {load_path}")
