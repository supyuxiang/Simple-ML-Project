"""
Core module for ML project.
Contains base classes, configuration management, logging, validation, and constants.
"""

from .interfaces import BaseModel, BaseDataProcessor, BaseTrainer, BaseEvaluator
from .config import ConfigManager
from .advanced_config import AdvancedConfigManager, ConfigSchema, SecretManager
from .logger import Logger
from .constants import (
    ModelType, ScalingMethod, MissingStrategy, ImputationMethod,
    DEFAULT_CONFIG, SUPPORTED_MODELS, SUPPORTED_METRICS,
    TARGET_COLUMN, ID_COLUMN, FEATURE_ENGINEERING, VALIDATION,
    OUTPUT_DIRS, LOGGING_CONFIG, ERROR_MESSAGES, SUCCESS_MESSAGES,
    PERFORMANCE_THRESHOLDS
)
from .exceptions import (
    MLProjectError, ConfigurationError, DataError, ModelError,
    TrainingError, EvaluationError, ValidationError, FileError,
    PreprocessingError, FeatureEngineeringError, ModelNotFittedError,
    InvalidDataError, EmptyDatasetError, MissingTargetError,
    UnsupportedModelError, UnsupportedMetricError, FileNotFoundError,
    ConfigurationValidationError, DataQualityError, TrainingConvergenceError,
    CrossValidationError, FeatureSelectionError, HyperparameterError,
    RegistryError, CacheError, TaskError, TaskTimeoutError, MonitoringError
)
from .validators import ConfigValidator, DataValidator, ModelValidator
from .registry import ComponentRegistry, register_component, get_component, list_components
from .cache import CacheManager, MemoryCacheBackend, DiskCacheBackend, cached, get_cache_manager, set_cache_backend
from .async_tasks import TaskScheduler, Task, FunctionTask, AsyncFunctionTask, submit_task, submit_function, start_scheduler, stop_scheduler
from .monitoring import MonitoringSystem, profile_function, add_metric, get_monitoring_system, start_monitoring, stop_monitoring

__all__ = [
    # Base classes
    'BaseModel',
    'BaseDataProcessor',
    'BaseTrainer',
    'BaseEvaluator',

    # Core utilities
    'ConfigManager',
    'AdvancedConfigManager',
    'ConfigSchema',
    'SecretManager',
    'Logger',

    # Constants and enums
    'ModelType',
    'ScalingMethod',
    'MissingStrategy',
    'ImputationMethod',
    'DEFAULT_CONFIG',
    'SUPPORTED_MODELS',
    'SUPPORTED_METRICS',
    'TARGET_COLUMN',
    'ID_COLUMN',
    'FEATURE_ENGINEERING',
    'VALIDATION',
    'OUTPUT_DIRS',
    'LOGGING_CONFIG',
    'ERROR_MESSAGES',
    'SUCCESS_MESSAGES',
    'PERFORMANCE_THRESHOLDS',

    # Exceptions
    'MLProjectError',
    'ConfigurationError',
    'DataError',
    'ModelError',
    'TrainingError',
    'EvaluationError',
    'ValidationError',
    'FileError',
    'PreprocessingError',
    'FeatureEngineeringError',
    'ModelNotFittedError',
    'InvalidDataError',
    'EmptyDatasetError',
    'MissingTargetError',
    'UnsupportedModelError',
    'UnsupportedMetricError',
    'FileNotFoundError',
    'ConfigurationValidationError',
    'DataQualityError',
    'TrainingConvergenceError',
    'CrossValidationError',
    'FeatureSelectionError',
    'HyperparameterError',
    'RegistryError',
    'CacheError',
    'TaskError',
    'TaskTimeoutError',
    'MonitoringError',

    # Validators
    'ConfigValidator',
    'DataValidator',
    'ModelValidator',

    # Registry system
    'ComponentRegistry',
    'register_component',
    'get_component',
    'list_components',

    # Caching system
    'CacheManager',
    'MemoryCacheBackend',
    'DiskCacheBackend',
    'cached',
    'get_cache_manager',
    'set_cache_backend',

    # Async task system
    'TaskScheduler',
    'Task',
    'FunctionTask',
    'AsyncFunctionTask',
    'submit_task',
    'submit_function',
    'start_scheduler',
    'stop_scheduler',

    # Monitoring system
    'MonitoringSystem',
    'profile_function',
    'add_metric',
    'get_monitoring_system',
    'start_monitoring',
    'stop_monitoring'
]
