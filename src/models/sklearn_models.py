"""
Scikit-learn models for ML1 project
Contains implementations of various sklearn models
"""

import numpy as np
from typing import Any, Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

from .base_model import BaseMLModel
from ..core.logger import Logger

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class LogisticRegressionModel(BaseMLModel):
    """
    Logistic Regression model for loan prediction
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        super().__init__(config, logger)
        self.model_name = "LogisticRegression"
    
    def build_model(self) -> LogisticRegression:
        """
        Build Logistic Regression model
        """
        default_params = {
            'random_state': 42,
            'max_iter': 1000,
            'solver': 'liblinear'
        }
        default_params.update(self.model_params)
        
        return LogisticRegression(**default_params)
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """
        Fit Logistic Regression model
        """
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


class RandomForestModel(BaseMLModel):
    """
    Random Forest model for loan prediction
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        super().__init__(config, logger)
        self.model_name = "RandomForest"
    
    def build_model(self) -> RandomForestClassifier:
        """
        Build Random Forest model
        """
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
        default_params.update(self.model_params)
        
        return RandomForestClassifier(**default_params)
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """
        Fit Random Forest model
        """
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


class XGBoostModel(BaseMLModel):
    """
    XGBoost model for loan prediction
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        super().__init__(config, logger)
        self.model_name = "XGBoost"
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install with: pip install xgboost")
    
    def build_model(self) -> xgb.XGBClassifier:
        """
        Build XGBoost model
        """
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        default_params.update(self.model_params)
        
        return xgb.XGBClassifier(**default_params)
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """
        Fit XGBoost model
        """
        if X_val is not None and y_val is not None:
            self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        else:
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


class LightGBMModel(BaseMLModel):
    """
    LightGBM model for loan prediction
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        super().__init__(config, logger)
        self.model_name = "LightGBM"
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Install with: pip install lightgbm")
    
    def build_model(self) -> lgb.LGBMClassifier:
        """
        Build LightGBM model
        """
        default_params = {
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbose': -1
        }
        default_params.update(self.model_params)
        
        return lgb.LGBMClassifier(**default_params)
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """
        Fit LightGBM model
        """
        if X_val is not None and y_val is not None:
            self.model.fit(X, y, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
        else:
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


class SVMModel(BaseMLModel):
    """
    Support Vector Machine model for loan prediction
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        super().__init__(config, logger)
        self.model_name = "SVM"
    
    def build_model(self) -> SVC:
        """
        Build SVM model
        """
        default_params = {
            'random_state': 42,
            'probability': True,
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale'
        }
        default_params.update(self.model_params)
        
        return SVC(**default_params)
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """
        Fit SVM model
        """
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


class NaiveBayesModel(BaseMLModel):
    """
    Naive Bayes model for loan prediction
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        super().__init__(config, logger)
        self.model_name = "NaiveBayes"
    
    def build_model(self) -> GaussianNB:
        """
        Build Naive Bayes model
        """
        default_params = {}
        default_params.update(self.model_params)
        
        return GaussianNB(**default_params)
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """
        Fit Naive Bayes model
        """
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


class KNNModel(BaseMLModel):
    """
    K-Nearest Neighbors model for loan prediction
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        super().__init__(config, logger)
        self.model_name = "KNN"
    
    def build_model(self) -> KNeighborsClassifier:
        """
        Build KNN model
        """
        default_params = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto'
        }
        default_params.update(self.model_params)
        
        return KNeighborsClassifier(**default_params)
    
    def _fit_model(self, X: np.ndarray, y: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """
        Fit KNN model
        """
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
