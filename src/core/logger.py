"""
Advanced logging system for ML project.

This module provides production-grade logging capabilities including:
- Multi-level logging with structured output
- File and console handlers with rotation
- Performance monitoring and metrics tracking
- Experiment tracking integration
- Error tracking and debugging support
"""

import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class Logger:
    """
    Production-grade logging system for ML projects.
    
    This class provides comprehensive logging capabilities including:
    - Multi-level structured logging
    - File rotation and archival
    - Performance metrics tracking
    - Experiment tracking integration
    - Error tracking and debugging
    - Custom formatters and filters
    
    Attributes:
        config (Dict[str, Any]): Logger configuration
        log_dir (Path): Directory for log files
        console_logger (logging.Logger): Console logger instance
        file_logger (logging.Logger): File logger instance
        experiment_logger (Optional[Any]): Experiment tracking logger
        performance_metrics (Dict[str, Any]): Performance tracking data
    """
    
    def __init__(self, config: Dict[str, Any], log_dir: Union[str, Path] = "logs") -> None:
        """
        Initialize the production logging system.
        
        Args:
            config: Logger configuration dictionary containing:
                - level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                - format: Log message format
                - file_handler: Enable file logging
                - console_handler: Enable console logging
                - rotation: File rotation settings
                - swanlab: Experiment tracking configuration
            log_dir: Directory to save log files
            
        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {
            'start_time': datetime.now(),
            'log_count': 0,
            'error_count': 0,
            'warning_count': 0
        }
        
        # Initialize loggers
        self._setup_console_logger()
        self._setup_file_logger()
        self._setup_experiment_logger()
        
        # Log initialization
        self.info("Production logging system initialized successfully")
        self.debug(f"Log directory: {self.log_dir.absolute()}")
        self.debug(f"Configuration: {self.config}")
    
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
        Setup file logging with rotation and archival.
        """
        self.file_logger = logging.getLogger('file')
        self.file_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.file_logger.handlers.clear()
        
        # Create log file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"ml_project_{timestamp}.log"
        
        # Get rotation settings from config
        rotation_config = self.config.get('rotation', {})
        max_bytes = rotation_config.get('max_bytes', 10 * 1024 * 1024)  # 10MB
        backup_count = rotation_config.get('backup_count', 5)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Create detailed formatter for file logs
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.file_logger.addHandler(file_handler)
        self.file_logger.propagate = False
        
        # Store log file path for reference
        self.log_file_path = log_file
    
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
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log debug message with optional context.
        
        Args:
            message: Debug message
            extra: Optional additional context data
        """
        self._log_with_metrics('debug', message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log info message with optional context.
        
        Args:
            message: Info message
            extra: Optional additional context data
        """
        self._log_with_metrics('info', message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log warning message with optional context.
        
        Args:
            message: Warning message
            extra: Optional additional context data
        """
        self._log_with_metrics('warning', message, extra)
        self.performance_metrics['warning_count'] += 1
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, 
              exc_info: bool = False) -> None:
        """
        Log error message with optional context and exception info.
        
        Args:
            message: Error message
            extra: Optional additional context data
            exc_info: Whether to include exception information
        """
        self._log_with_metrics('error', message, extra, exc_info)
        self.performance_metrics['error_count'] += 1
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None,
                 exc_info: bool = False) -> None:
        """
        Log critical message with optional context and exception info.
        
        Args:
            message: Critical message
            extra: Optional additional context data
            exc_info: Whether to include exception information
        """
        self._log_with_metrics('critical', message, extra, exc_info)
        self.performance_metrics['error_count'] += 1
    
    def _log_with_metrics(self, level: str, message: str, 
                         extra: Optional[Dict[str, Any]] = None,
                         exc_info: bool = False) -> None:
        """
        Internal method to log with performance tracking.
        
        Args:
            level: Log level
            message: Log message
            extra: Optional context data
            exc_info: Whether to include exception info
        """
        # Update performance metrics
        self.performance_metrics['log_count'] += 1
        
        # Format message with context if provided
        if extra:
            context_str = ', '.join([f"{k}={v}" for k, v in extra.items()])
            message = f"{message} | Context: {context_str}"
        
        # Log to console and file
        getattr(self.console_logger, level)(message, exc_info=exc_info)
        getattr(self.file_logger, level)(message, exc_info=exc_info)
        
        # Log to experiment tracker if available
        if self.experiment_logger and level in ['info', 'warning', 'error', 'critical']:
            try:
                self.experiment_logger.log({
                    f"log_{level}": message,
                    "log_count": self.performance_metrics['log_count']
                })
            except Exception as e:
                # Avoid recursive logging
                print(f"Failed to log to experiment tracker: {e}")
    
    def log_exception(self, message: str, exception: Exception, 
                     extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log exception with full traceback and context.
        
        Args:
            message: Error message
            exception: Exception instance
            extra: Optional additional context data
        """
        error_context = {
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'traceback': traceback.format_exc()
        }
        
        if extra:
            error_context.update(extra)
        
        self.error(f"{message}: {exception}", extra=error_context, exc_info=True)
    
    def log_performance_metrics(self) -> None:
        """
        Log current performance metrics.
        """
        current_time = datetime.now()
        runtime = (current_time - self.performance_metrics['start_time']).total_seconds()
        
        metrics = {
            'runtime_seconds': runtime,
            'log_count': self.performance_metrics['log_count'],
            'error_count': self.performance_metrics['error_count'],
            'warning_count': self.performance_metrics['warning_count'],
            'logs_per_second': self.performance_metrics['log_count'] / max(runtime, 1)
        }
        
        self.info("Performance Metrics", extra=metrics)
        
        if self.experiment_logger:
            try:
                self.experiment_logger.log({"performance_metrics": metrics})
            except Exception as e:
                self.warning(f"Failed to log performance metrics: {e}")
    
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
