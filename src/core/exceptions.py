"""
Custom exceptions for the ML project.
This module defines all custom exceptions used throughout the project.
"""

from typing import Optional, Any


class MLProjectError(Exception):
    """Base exception class for ML project errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Any] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            error_code: Optional error code
            details: Optional additional details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details
    
    def __str__(self) -> str:
        """String representation of the exception."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(MLProjectError):
    """Raised when there's a configuration-related error."""
    pass


class DataError(MLProjectError):
    """Raised when there's a data-related error."""
    pass


class ModelError(MLProjectError):
    """Raised when there's a model-related error."""
    pass


class TrainingError(MLProjectError):
    """Raised when there's a training-related error."""
    pass


class EvaluationError(MLProjectError):
    """Raised when there's an evaluation-related error."""
    pass


class ValidationError(MLProjectError):
    """Raised when there's a validation-related error."""
    pass


class FileError(MLProjectError):
    """Raised when there's a file I/O related error."""
    pass


class PreprocessingError(MLProjectError):
    """Raised when there's a preprocessing-related error."""
    pass


class FeatureEngineeringError(MLProjectError):
    """Raised when there's a feature engineering-related error."""
    pass


class ModelNotFittedError(ModelError):
    """Raised when trying to use a model that hasn't been fitted."""
    
    def __init__(self, model_name: str = "Model"):
        """
        Initialize the exception.
        
        Args:
            model_name: Name of the model
        """
        message = f"{model_name} must be fitted before making predictions"
        super().__init__(message, error_code="MODEL_NOT_FITTED")


class InvalidDataError(DataError):
    """Raised when data is invalid or malformed."""
    
    def __init__(self, message: str = "Invalid data provided", details: Optional[Any] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Optional additional details
        """
        super().__init__(message, error_code="INVALID_DATA", details=details)


class EmptyDatasetError(DataError):
    """Raised when dataset is empty."""
    
    def __init__(self, message: str = "Dataset is empty"):
        """
        Initialize the exception.
        
        Args:
            message: Error message
        """
        super().__init__(message, error_code="EMPTY_DATASET")


class MissingTargetError(DataError):
    """Raised when target column is missing from data."""
    
    def __init__(self, target_name: str = "target"):
        """
        Initialize the exception.
        
        Args:
            target_name: Name of the missing target column
        """
        message = f"Target column '{target_name}' not found in data"
        super().__init__(message, error_code="MISSING_TARGET")


class UnsupportedModelError(ModelError):
    """Raised when an unsupported model is requested."""
    
    def __init__(self, model_name: str):
        """
        Initialize the exception.
        
        Args:
            model_name: Name of the unsupported model
        """
        message = f"Unsupported model type: {model_name}"
        super().__init__(message, error_code="INVALID_MODEL")


class UnsupportedMetricError(EvaluationError):
    """Raised when an unsupported metric is requested."""
    
    def __init__(self, metric_name: str):
        """
        Initialize the exception.
        
        Args:
            metric_name: Name of the unsupported metric
        """
        message = f"Unsupported metric: {metric_name}"
        super().__init__(message, error_code="INVALID_METRIC")


class FileNotFoundError(FileError):
    """Raised when a required file is not found."""
    
    def __init__(self, file_path: str):
        """
        Initialize the exception.
        
        Args:
            file_path: Path to the missing file
        """
        message = f"File not found: {file_path}"
        super().__init__(message, error_code="FILE_NOT_FOUND")


class ConfigurationValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    
    def __init__(self, message: str, details: Optional[Any] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Optional additional details
        """
        super().__init__(message, error_code="CONFIG_VALIDATION_FAILED", details=details)


class DataQualityError(DataError):
    """Raised when data quality is below acceptable threshold."""
    
    def __init__(self, quality_score: float, threshold: float = 0.5):
        """
        Initialize the exception.
        
        Args:
            quality_score: Actual quality score
            threshold: Minimum acceptable quality score
        """
        message = f"Data quality score {quality_score:.2f} is below threshold {threshold:.2f}"
        super().__init__(message, error_code="DATA_QUALITY_LOW", details={
            "quality_score": quality_score,
            "threshold": threshold
        })


class TrainingConvergenceError(TrainingError):
    """Raised when model training fails to converge."""
    
    def __init__(self, model_name: str, max_iterations: int):
        """
        Initialize the exception.
        
        Args:
            model_name: Name of the model
            max_iterations: Maximum iterations reached
        """
        message = f"{model_name} training failed to converge after {max_iterations} iterations"
        super().__init__(message, error_code="TRAINING_CONVERGENCE_FAILED", details={
            "model_name": model_name,
            "max_iterations": max_iterations
        })


class CrossValidationError(EvaluationError):
    """Raised when cross-validation fails."""
    
    def __init__(self, message: str = "Cross-validation failed", details: Optional[Any] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Optional additional details
        """
        super().__init__(message, error_code="CROSS_VALIDATION_FAILED", details=details)


class FeatureSelectionError(FeatureEngineeringError):
    """Raised when feature selection fails."""
    
    def __init__(self, message: str = "Feature selection failed", details: Optional[Any] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Optional additional details
        """
        super().__init__(message, error_code="FEATURE_SELECTION_FAILED", details=details)


class HyperparameterError(ModelError):
    """Raised when hyperparameter optimization fails."""
    
    def __init__(self, message: str = "Hyperparameter optimization failed", details: Optional[Any] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Optional additional details
        """
        super().__init__(message, error_code="HYPERPARAMETER_OPTIMIZATION_FAILED", details=details)


# Registry and Component Errors
class RegistryError(MLProjectError):
    """Exception raised for registry-related errors."""
    pass


# Cache Errors
class CacheError(MLProjectError):
    """Exception raised for cache-related errors."""
    pass


# Task and Async Errors
class TaskError(MLProjectError):
    """Exception raised for task-related errors."""
    pass


class TaskTimeoutError(TaskError):
    """Exception raised when a task times out."""
    pass


# Monitoring Errors
class MonitoringError(MLProjectError):
    """Exception raised for monitoring-related errors."""
    pass
