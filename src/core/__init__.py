"""
Core module for ML1 project
Contains base classes and interfaces for the machine learning pipeline
"""

from .base import BaseModel, BaseDataProcessor, BaseTrainer, BaseEvaluator
from .config import ConfigManager
from .logger import Logger

__all__ = [
    'BaseModel',
    'BaseDataProcessor', 
    'BaseTrainer',
    'BaseEvaluator',
    'ConfigManager',
    'Logger'
]
