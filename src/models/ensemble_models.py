"""
Ensemble models for ML1 project
Contains voting, stacking, and bagging ensemble implementations
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
from sklearn.ensemble import VotingClassifier, BaggingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

from .base_model import BaseMLModel
from ..core.logger import Logger


class VotingEnsembleModel(BaseMLModel):
    """
    Voting ensemble model for loan prediction
    Combines multiple models using majority voting
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        """
        Initialize the voting ensemble model
        
        Args:
            config: Model configuration
            logger: Logger instance
        """
        super().__init__(config, logger)
        self.model_name = "VotingEnsemble"
        self.base_models = []
        self.ensemble_type = config.get('ensemble_type', 'hard')  # 'hard' or 'soft'
        
        # Initialize base models
        self._initialize_base_models(config)
    
    def _initialize_base_models(self, config: Dict[str, Any]) -> None:
        """
        Initialize base models for ensemble
        
        Args:
            config: Model configuration
        """
        from .sklearn_models import (
            LogisticRegressionModel, RandomForestModel, 
            SVMModel, NaiveBayesModel
        )
        
        base_model_configs = config.get('base_models', [
            {
                'model_name': 'LogisticRegression',
                'model_type': 'classification',
                'model_params': {'random_state': 42, 'max_iter': 1000}
            },
            {
                'model_name': 'RandomForest',
                'model_type': 'classification',
                'model_params': {'n_estimators': 50, 'random_state': 42}
            },
            {
                'model_name': 'SVM',
                'model_type': 'classification',
                'model_params': {'random_state': 42, 'probability': True}
            },
            {
                'model_name': 'NaiveBayes',
                'model_type': 'classification',
                'model_params': {}
            }
        ])
        
        for model_config in base_model_configs:
            try:
                if model_config['model_name'] == 'LogisticRegression':
                    model = LogisticRegressionModel(model_config, self.logger)
                elif model_config['model_name'] == 'RandomForest':
                    model = RandomForestModel(model_config, self.logger)
                elif model_config['model_name'] == 'SVM':
                    model = SVMModel(model_config, self.logger)
                elif model_config['model_name'] == 'NaiveBayes':
                    model = NaiveBayesModel(model_config, self.logger)
                else:
                    continue
                
                self.base_models.append(model)
                
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Could not initialize {model_config['model_name']}: {e}")
    
    def build_model(self) -> VotingClassifier:
        """
        Build voting ensemble model
        """
        # Create estimators list for VotingClassifier
        estimators = []
        for i, base_model in enumerate(self.base_models):
            base_model.model = base_model.build_model()
            estimators.append((f"model_{i}", base_model.model))
        
        voting_type = 'hard' if self.ensemble_type == 'hard' else 'soft'
        
        return VotingClassifier(
            estimators=estimators,
            voting=voting_type,
            n_jobs=-1
        )
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """
        Fit voting ensemble model
        """
        # Train individual base models first
        for base_model in self.base_models:
            base_model.fit(X, y, X_val, y_val)
        
        # Train the ensemble
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        """
        return self.model.predict(X)
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities
        """
        return self.model.predict_proba(X)


class StackingEnsembleModel(BaseMLModel):
    """
    Stacking ensemble model for loan prediction
    Uses meta-learner to combine base model predictions
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        """
        Initialize the stacking ensemble model
        
        Args:
            config: Model configuration
            logger: Logger instance
        """
        super().__init__(config, logger)
        self.model_name = "StackingEnsemble"
        self.base_models = []
        self.meta_learner = config.get('meta_learner', 'LogisticRegression')
        self.cv_folds = config.get('cv_folds', 5)
        
        # Initialize base models
        self._initialize_base_models(config)
    
    def _initialize_base_models(self, config: Dict[str, Any]) -> None:
        """
        Initialize base models for stacking
        
        Args:
            config: Model configuration
        """
        from .sklearn_models import (
            LogisticRegressionModel, RandomForestModel, 
            SVMModel, NaiveBayesModel, KNNModel
        )
        
        base_model_configs = config.get('base_models', [
            {
                'model_name': 'LogisticRegression',
                'model_type': 'classification',
                'model_params': {'random_state': 42, 'max_iter': 1000}
            },
            {
                'model_name': 'RandomForest',
                'model_type': 'classification',
                'model_params': {'n_estimators': 50, 'random_state': 42}
            },
            {
                'model_name': 'SVM',
                'model_type': 'classification',
                'model_params': {'random_state': 42, 'probability': True}
            },
            {
                'model_name': 'NaiveBayes',
                'model_type': 'classification',
                'model_params': {}
            },
            {
                'model_name': 'KNN',
                'model_type': 'classification',
                'model_params': {'n_neighbors': 5}
            }
        ])
        
        for model_config in base_model_configs:
            try:
                if model_config['model_name'] == 'LogisticRegression':
                    model = LogisticRegressionModel(model_config, self.logger)
                elif model_config['model_name'] == 'RandomForest':
                    model = RandomForestModel(model_config, self.logger)
                elif model_config['model_name'] == 'SVM':
                    model = SVMModel(model_config, self.logger)
                elif model_config['model_name'] == 'NaiveBayes':
                    model = NaiveBayesModel(model_config, self.logger)
                elif model_config['model_name'] == 'KNN':
                    model = KNNModel(model_config, self.logger)
                else:
                    continue
                
                self.base_models.append(model)
                
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Could not initialize {model_config['model_name']}: {e}")
    
    def build_model(self) -> StackingClassifier:
        """
        Build stacking ensemble model
        """
        # Create estimators list for StackingClassifier
        estimators = []
        for i, base_model in enumerate(self.base_models):
            base_model.model = base_model.build_model()
            estimators.append((f"model_{i}", base_model.model))
        
        # Create meta-learner
        if self.meta_learner == 'LogisticRegression':
            meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        else:
            meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        return StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=self.cv_folds,
            n_jobs=-1
        )
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """
        Fit stacking ensemble model
        """
        # Train individual base models first
        for base_model in self.base_models:
            base_model.fit(X, y, X_val, y_val)
        
        # Train the stacking ensemble
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        """
        return self.model.predict(X)
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities
        """
        return self.model.predict_proba(X)


class BaggingEnsembleModel(BaseMLModel):
    """
    Bagging ensemble model for loan prediction
    Uses bootstrap aggregating with a single base model
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        """
        Initialize the bagging ensemble model
        
        Args:
            config: Model configuration
            logger: Logger instance
        """
        super().__init__(config, logger)
        self.model_name = "BaggingEnsemble"
        self.base_model_name = config.get('base_model', 'RandomForest')
        self.n_estimators = config.get('n_estimators', 10)
        self.max_samples = config.get('max_samples', 1.0)
        self.max_features = config.get('max_features', 1.0)
        self.bootstrap = config.get('bootstrap', True)
        self.bootstrap_features = config.get('bootstrap_features', False)
        
        # Initialize base model
        self._initialize_base_model(config)
    
    def _initialize_base_model(self, config: Dict[str, Any]) -> None:
        """
        Initialize base model for bagging
        
        Args:
            config: Model configuration
        """
        from .sklearn_models import (
            LogisticRegressionModel, RandomForestModel, 
            SVMModel, NaiveBayesModel, KNNModel
        )
        
        base_model_config = {
            'model_name': self.base_model_name,
            'model_type': 'classification',
            'model_params': config.get('base_model_params', {'random_state': 42})
        }
        
        try:
            if self.base_model_name == 'LogisticRegression':
                self.base_model = LogisticRegressionModel(base_model_config, self.logger)
            elif self.base_model_name == 'RandomForest':
                self.base_model = RandomForestModel(base_model_config, self.logger)
            elif self.base_model_name == 'SVM':
                self.base_model = SVMModel(base_model_config, self.logger)
            elif self.base_model_name == 'NaiveBayes':
                self.base_model = NaiveBayesModel(base_model_config, self.logger)
            elif self.base_model_name == 'KNN':
                self.base_model = KNNModel(base_model_config, self.logger)
            else:
                # Default to RandomForest
                self.base_model = RandomForestModel(base_model_config, self.logger)
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Could not initialize {self.base_model_name}: {e}")
            # Fallback to RandomForest
            self.base_model = RandomForestModel(base_model_config, self.logger)
    
    def build_model(self) -> BaggingClassifier:
        """
        Build bagging ensemble model
        """
        # Build base model
        base_estimator = self.base_model.build_model()
        
        return BaggingClassifier(
            base_estimator=base_estimator,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            bootstrap_features=self.bootstrap_features,
            random_state=42,
            n_jobs=-1
        )
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """
        Fit bagging ensemble model
        """
        # Train the base model first (for feature importance calculation)
        self.base_model.fit(X, y, X_val, y_val)
        
        # Train the bagging ensemble
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        """
        return self.model.predict(X)
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities
        """
        return self.model.predict_proba(X)
    
    def get_base_model_performance(self) -> Dict[str, Any]:
        """
        Get performance of individual base models
        
        Returns:
            Dictionary containing base model performance
        """
        if not hasattr(self.model, 'estimators_'):
            return {}
        
        base_scores = []
        for estimator in self.model.estimators_:
            # This would require the original training data
            # For now, return empty dict
            pass
        
        return {
            'n_estimators': len(self.model.estimators_),
            'base_model_name': self.base_model_name,
            'bagging_params': {
                'max_samples': self.max_samples,
                'max_features': self.max_features,
                'bootstrap': self.bootstrap
            }
        }
