"""
Logging system for ML1 project
Provides comprehensive logging capabilities with multiple outputs
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


class Logger:
    """
    Comprehensive logging system for the ML1 project
    Supports console, file, and experiment tracking logging
    """
    
    def __init__(self, config: Dict[str, Any], log_dir: str = "logs"):
        """
        Initialize the logger
        
        Args:
            config: Logger configuration
            log_dir: Directory to save log files
        """
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize loggers
        self._setup_console_logger()
        self._setup_file_logger()
        self._setup_experiment_logger()
        
        # Log initialization
        self.info("Logger initialized successfully")
    
    def _setup_console_logger(self) -> None:
        """
        Setup console logging
        """
        self.console_logger = logging.getLogger('console')
        self.console_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.console_logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.console_logger.addHandler(console_handler)
        self.console_logger.propagate = False
    
    def _setup_file_logger(self) -> None:
        """
        Setup file logging
        """
        self.file_logger = logging.getLogger('file')
        self.file_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.file_logger.handlers.clear()
        
        # Create log file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"ml1_{timestamp}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.file_logger.addHandler(file_handler)
        self.file_logger.propagate = False
    
    def _setup_experiment_logger(self) -> None:
        """
        Setup experiment tracking logger (SwanLab)
        """
        self.experiment_logger = None
        
        if self.config.get('swanlab', {}).get('open', False):
            try:
                import swanlab
                
                # Initialize SwanLab
                swanlab.init(
                    project=self.config['swanlab'].get('project', 'ML1'),
                    experiment_name=self.config['swanlab'].get('name', 'experiment'),
                    config=self.config
                )
                
                self.experiment_logger = swanlab
                self.info("SwanLab experiment tracking initialized")
                
            except ImportError:
                self.warning("SwanLab not available. Install with: pip install swanlab")
            except Exception as e:
                self.warning(f"Failed to initialize SwanLab: {e}")
    
    def debug(self, message: str) -> None:
        """
        Log debug message
        
        Args:
            message: Debug message
        """
        self.console_logger.debug(message)
        self.file_logger.debug(message)
    
    def info(self, message: str) -> None:
        """
        Log info message
        
        Args:
            message: Info message
        """
        self.console_logger.info(message)
        self.file_logger.info(message)
    
    def warning(self, message: str) -> None:
        """
        Log warning message
        
        Args:
            message: Warning message
        """
        self.console_logger.warning(message)
        self.file_logger.warning(message)
    
    def error(self, message: str) -> None:
        """
        Log error message
        
        Args:
            message: Error message
        """
        self.console_logger.error(message)
        self.file_logger.error(message)
    
    def critical(self, message: str) -> None:
        """
        Log critical message
        
        Args:
            message: Critical message
        """
        self.console_logger.critical(message)
        self.file_logger.critical(message)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to experiment tracker
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number (optional)
        """
        if self.experiment_logger:
            try:
                if step is not None:
                    self.experiment_logger.log(metrics, step=step)
                else:
                    self.experiment_logger.log(metrics)
            except Exception as e:
                self.warning(f"Failed to log metrics: {e}")
    
    def log_model_info(self, model_info: Dict[str, Any]) -> None:
        """
        Log model information
        
        Args:
            model_info: Dictionary containing model information
        """
        self.info("Model Information:")
        for key, value in model_info.items():
            self.info(f"  {key}: {value}")
        
        if self.experiment_logger:
            try:
                self.experiment_logger.log({"model_info": model_info})
            except Exception as e:
                self.warning(f"Failed to log model info: {e}")
    
    def log_data_info(self, data_info: Dict[str, Any]) -> None:
        """
        Log data information
        
        Args:
            data_info: Dictionary containing data information
        """
        self.info("Data Information:")
        for key, value in data_info.items():
            self.info(f"  {key}: {value}")
        
        if self.experiment_logger:
            try:
                self.experiment_logger.log({"data_info": data_info})
            except Exception as e:
                self.warning(f"Failed to log data info: {e}")
    
    def log_training_progress(self, epoch: int, train_loss: float, 
                            val_loss: Optional[float] = None, 
                            val_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Log training progress
        
        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss (optional)
            val_metrics: Validation metrics (optional)
        """
        message = f"Epoch {epoch}: Train Loss = {train_loss:.4f}"
        if val_loss is not None:
            message += f", Val Loss = {val_loss:.4f}"
        
        self.info(message)
        
        # Log to experiment tracker
        metrics = {"epoch": epoch, "train_loss": train_loss}
        if val_loss is not None:
            metrics["val_loss"] = val_loss
        if val_metrics:
            metrics.update(val_metrics)
        
        self.log_metrics(metrics, step=epoch)
    
    def log_evaluation_results(self, results: Dict[str, Any]) -> None:
        """
        Log evaluation results
        
        Args:
            results: Dictionary containing evaluation results
        """
        self.info("Evaluation Results:")
        for key, value in results.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")
        
        if self.experiment_logger:
            try:
                self.experiment_logger.log({"evaluation": results})
            except Exception as e:
                self.warning(f"Failed to log evaluation results: {e}")
    
    def close(self) -> None:
        """
        Close the logger and finish experiment tracking
        """
        if self.experiment_logger:
            try:
                self.experiment_logger.finish()
                self.info("Experiment tracking finished")
            except Exception as e:
                self.warning(f"Failed to finish experiment tracking: {e}")
        
        self.info("Logger closed")
