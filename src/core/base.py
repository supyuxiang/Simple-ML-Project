"""
Base classes for the ML1 project
Implements the core interfaces and abstract base classes
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model with configuration
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        
    @abstractmethod
    def build_model(self) -> Any:
        """
        Build the model architecture
        
        Returns:
            The built model
        """
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """
        Train the model
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features to predict
            
        Returns:
            Class probabilities
        """
        pass
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        # Implementation depends on the specific model type
        pass
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model
        
        Args:
            path: Path to load the model from
        """
        # Implementation depends on the specific model type
        pass


class BaseDataProcessor(ABC):
    """
    Abstract base class for data processing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data processor
        
        Args:
            config: Data processing configuration
        """
        self.config = config
        self.feature_names = None
        self.target_name = None
        self.preprocessors = {}
        
    @abstractmethod
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from file
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        pass
    
    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data
        
        Args:
            data: Raw data
            
        Returns:
            Tuple of (features, labels)
        """
        pass
    
    @abstractmethod
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and validation sets
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        pass


class BaseTrainer(ABC):
    """
    Abstract base class for model training
    """
    
    def __init__(self, model: BaseModel, config: Dict[str, Any]):
        """
        Initialize the trainer
        
        Args:
            model: Model to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.training_history = {}
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training history
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Evaluation metrics
        """
        pass


class BaseEvaluator(ABC):
    """
    Abstract base class for model evaluation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.metrics = {}
        
    @abstractmethod
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
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
        pass
    
    @abstractmethod
    def plot_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    y_proba: Optional[np.ndarray] = None, save_path: Optional[str] = None) -> None:
        """
        Plot evaluation results
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            save_path: Path to save the plot (optional)
        """
        pass
