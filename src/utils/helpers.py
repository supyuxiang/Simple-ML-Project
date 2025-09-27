"""
Helper functions for ML1 project
Contains utility functions for common operations
"""

import os
import json
import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib


def setup_directories(base_path: str, directories: List[str]) -> None:
    """
    Setup required directories
    
    Args:
        base_path: Base directory path
        directories: List of directory names to create
    """
    base_path = Path(base_path)
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)


def save_results(results: Dict[str, Any], file_path: str, format: str = 'yaml') -> None:
    """
    Save results to file
    
    Args:
        results: Results dictionary
        file_path: Path to save the file
        format: File format ('yaml', 'json', 'pickle')
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'yaml':
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
    
    elif format.lower() == 'json':
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    elif format.lower() == 'pickle':
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(file_path: str, format: str = 'yaml') -> Dict[str, Any]:
    """
    Load results from file
    
    Args:
        file_path: Path to the file
        format: File format ('yaml', 'json', 'pickle')
        
    Returns:
        Loaded results dictionary
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if format.lower() == 'yaml':
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    elif format.lower() == 'json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    elif format.lower() == 'pickle':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_model(model: Any, file_path: str) -> None:
    """
    Save a trained model
    
    Args:
        model: Trained model object
        file_path: Path to save the model
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, file_path)


def load_model(file_path: str) -> Any:
    """
    Load a trained model
    
    Args:
        file_path: Path to the model file
        
    Returns:
        Loaded model object
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    return joblib.load(file_path)


def create_submission_file(predictions: np.ndarray, sample_ids: np.ndarray, 
                          file_path: str) -> None:
    """
    Create a submission file for competitions
    
    Args:
        predictions: Model predictions
        sample_ids: Sample IDs
        file_path: Path to save the submission file
    """
    submission_df = pd.DataFrame({
        'ID': sample_ids,
        'Prediction': predictions
    })
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    submission_df.to_csv(file_path, index=False)


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        y: Target labels
        
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return dict(zip(classes, weights))


def get_feature_importance_ranking(feature_importance: Dict[str, float], 
                                 top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Get top K most important features
    
    Args:
        feature_importance: Dictionary of feature importance scores
        top_k: Number of top features to return
        
    Returns:
        List of (feature_name, importance_score) tuples
    """
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    return sorted_features[:top_k]


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, str]:
    """
    Format metrics for display
    
    Args:
        metrics: Dictionary of metrics
        precision: Number of decimal places
        
    Returns:
        Dictionary of formatted metrics
    """
    return {key: f"{value:.{precision}f}" for key, value in metrics.items()}


def create_summary_table(model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a summary table of model results
    
    Args:
        model_results: Dictionary of model results
        
    Returns:
        Summary DataFrame
    """
    summary_data = []
    
    for model_name, results in model_results.items():
        if 'error' not in results:
            test_metrics = results.get('test_metrics', {})
            row = {'Model': model_name}
            row.update(test_metrics)
            summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def validate_data_format(data: Union[np.ndarray, pd.DataFrame], 
                        expected_shape: Optional[tuple] = None) -> bool:
    """
    Validate data format and shape
    
    Args:
        data: Data to validate
        expected_shape: Expected shape (optional)
        
    Returns:
        True if valid, False otherwise
    """
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = data
    
    if not isinstance(data_array, np.ndarray):
        return False
    
    if expected_shape is not None:
        if data_array.shape != expected_shape:
            return False
    
    if np.isnan(data_array).any():
        return False
    
    return True


def get_memory_usage(data: Union[np.ndarray, pd.DataFrame]) -> float:
    """
    Get memory usage of data in MB
    
    Args:
        data: Data to measure
        
    Returns:
        Memory usage in MB
    """
    if isinstance(data, pd.DataFrame):
        return data.memory_usage(deep=True).sum() / 1024**2
    else:
        return data.nbytes / 1024**2


def print_system_info() -> None:
    """
    Print system information
    """
    import platform
    import sys
    import psutil
    
    print("System Information:")
    print(f"  Platform: {platform.platform()}")
    print(f"  Python Version: {sys.version}")
    print(f"  CPU Count: {psutil.cpu_count()}")
    print(f"  Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    # Set TensorFlow seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
