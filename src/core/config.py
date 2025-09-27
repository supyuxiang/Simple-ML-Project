"""
Configuration management for ML1 project
Handles loading and validation of configuration files
"""

import yaml
import os
from typing import Any, Dict, Optional
from pathlib import Path


class ConfigManager:
    """
    Configuration manager for the ML1 project
    Handles loading, validation, and access to configuration parameters
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path or "config.yaml"
        self.config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """
        Load configuration from YAML file
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
            self._validate_config()
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def _validate_config(self) -> None:
        """
        Validate the configuration structure
        """
        required_sections = ['Model', 'Data', 'Train', 'Metrics']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'Model.model_name')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'Model.model_name')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file
        
        Args:
            path: Path to save the configuration (optional)
        """
        save_path = path or self.config_path
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(self.config, file, default_flow_style=False, allow_unicode=True)
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration
        
        Returns:
            Model configuration dictionary
        """
        return self.get('Model', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data configuration
        
        Returns:
            Data configuration dictionary
        """
        return self.get('Data', {})
    
    def get_train_config(self) -> Dict[str, Any]:
        """
        Get training configuration
        
        Returns:
            Training configuration dictionary
        """
        return self.get('Train', {})
    
    def get_metrics_config(self) -> Dict[str, Any]:
        """
        Get metrics configuration
        
        Returns:
            Metrics configuration dictionary
        """
        return self.get('Metrics', {})
    
    def get_logger_config(self) -> Dict[str, Any]:
        """
        Get logger configuration
        
        Returns:
            Logger configuration dictionary
        """
        return self.get('Logger', {})
    
    def get_device_config(self) -> Dict[str, Any]:
        """
        Get device configuration
        
        Returns:
            Device configuration dictionary
        """
        return self.get('Device', {})
    
    def update_paths(self, base_path: str) -> None:
        """
        Update all file paths in configuration with base path
        
        Args:
            base_path: Base directory path
        """
        base_path = Path(base_path).resolve()
        
        # Update model save paths
        model_save_dir = self.get('Train.save.model.save_dir')
        if model_save_dir:
            self.set('Train.save.model.save_dir', str(base_path / 'outputs' / 'models'))
        
        # Update curve save paths
        curve_save_dir = self.get('Train.save.curve.save_dir')
        if curve_save_dir:
            self.set('Train.save.curve.save_dir', str(base_path / 'outputs' / 'curves'))
        
        # Update report save paths
        report_save_dir = self.get('Train.save.report.save_dir')
        if report_save_dir:
            self.set('Train.save.report.save_dir', str(base_path / 'outputs' / 'reports'))
    
    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value
        """
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        Allow dictionary-style setting
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """
        Check if key exists in configuration
        
        Args:
            key: Configuration key
            
        Returns:
            True if key exists, False otherwise
        """
        return self.get(key) is not None
