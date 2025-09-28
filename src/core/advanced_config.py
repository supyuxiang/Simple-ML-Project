"""
Advanced configuration management with environment-specific settings,
validation, and dynamic updates.

This module provides:
- Environment-specific configuration overrides
- Configuration validation and schema enforcement
- Dynamic configuration updates
- Configuration versioning and migration
- Secret management and encryption
"""

import os
import json
import yaml
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, Callable
from datetime import datetime
import threading
from cryptography.fernet import Fernet
import base64

from .exceptions import ConfigurationError, ConfigurationValidationError
from .logger import Logger


@dataclass
class ConfigSchema:
    """Configuration schema definition."""
    name: str
    version: str
    fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required_fields: List[str] = field(default_factory=list)
    validators: Dict[str, Callable] = field(default_factory=dict)
    
    def validate(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate configuration against schema."""
        errors = {}
        
        # Check required fields
        for field in self.required_fields:
            if field not in config:
                errors.setdefault('missing_fields', []).append(field)
        
        # Validate field types and constraints
        for field_name, field_config in self.fields.items():
            if field_name in config:
                value = config[field_name]
                
                # Type validation
                expected_type = field_config.get('type')
                if expected_type and not isinstance(value, expected_type):
                    errors.setdefault('type_errors', []).append(
                        f"{field_name}: expected {expected_type.__name__}, got {type(value).__name__}"
                    )
                
                # Range validation
                if 'min' in field_config and value < field_config['min']:
                    errors.setdefault('range_errors', []).append(
                        f"{field_name}: value {value} is below minimum {field_config['min']}"
                    )
                
                if 'max' in field_config and value > field_config['max']:
                    errors.setdefault('range_errors', []).append(
                        f"{field_name}: value {value} is above maximum {field_config['max']}"
                    )
                
                # Custom validation
                if field_name in self.validators:
                    try:
                        self.validators[field_name](value)
                    except Exception as e:
                        errors.setdefault('validation_errors', []).append(
                            f"{field_name}: {str(e)}"
                        )
        
        return errors


class SecretManager:
    """Manages encrypted secrets and sensitive configuration."""
    
    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            key = Fernet.generate_key()
        self.cipher = Fernet(key)
        self._secrets: Dict[str, str] = {}
    
    def encrypt_secret(self, value: str) -> str:
        """Encrypt a secret value."""
        encrypted = self.cipher.encrypt(value.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_secret(self, encrypted_value: str) -> str:
        """Decrypt a secret value."""
        encrypted = base64.b64decode(encrypted_value.encode())
        decrypted = self.cipher.decrypt(encrypted)
        return decrypted.decode()
    
    def store_secret(self, key: str, value: str) -> None:
        """Store an encrypted secret."""
        self._secrets[key] = self.encrypt_secret(value)
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get and decrypt a secret."""
        if key in self._secrets:
            return self.decrypt_secret(self._secrets[key])
        return None


class ConfigWatcher:
    """Watches configuration files for changes."""
    
    def __init__(self, callback: Callable[[str], None], logger: Optional[Logger] = None):
        self.callback = callback
        self.logger = logger
        self._watched_files: Dict[str, float] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def watch_file(self, filepath: Union[str, Path]) -> None:
        """Start watching a configuration file."""
        filepath = Path(filepath)
        if filepath.exists():
            self._watched_files[str(filepath)] = filepath.stat().st_mtime
    
    def start_watching(self, interval: float = 1.0) -> None:
        """Start the file watcher."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._watch_loop,
            args=(interval,),
            daemon=True
        )
        self._thread.start()
        
        if self.logger:
            self.logger.info("Configuration watcher started")
    
    def stop_watching(self) -> None:
        """Stop the file watcher."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        
        if self.logger:
            self.logger.info("Configuration watcher stopped")
    
    def _watch_loop(self, interval: float) -> None:
        """Main watching loop."""
        import time
        
        while self._running:
            try:
                for filepath_str, last_mtime in list(self._watched_files.items()):
                    filepath = Path(filepath_str)
                    if filepath.exists():
                        current_mtime = filepath.stat().st_mtime
                        if current_mtime > last_mtime:
                            self._watched_files[filepath_str] = current_mtime
                            self.callback(filepath_str)
                            if self.logger:
                                self.logger.info(f"Configuration file changed: {filepath}")
                
                time.sleep(interval)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in config watcher: {e}")
                time.sleep(interval)


class AdvancedConfigManager:
    """
    Advanced configuration manager with production features.
    
    Features:
    - Environment-specific overrides
    - Configuration validation
    - Secret management
    - Dynamic updates
    - Configuration versioning
    - File watching
    """
    
    def __init__(
        self,
        config_path: Union[str, Path],
        environment: str = "development",
        schema: Optional[ConfigSchema] = None,
        logger: Optional[Logger] = None
    ):
        self.config_path = Path(config_path)
        self.environment = environment
        self.schema = schema
        self.logger = logger
        
        # Configuration storage
        self._base_config: Dict[str, Any] = {}
        self._env_config: Dict[str, Any] = {}
        self._merged_config: Dict[str, Any] = {}
        self._config_history: List[Dict[str, Any]] = []
        
        # Advanced features
        self._secret_manager = SecretManager()
        self._watcher: Optional[ConfigWatcher] = None
        self._lock = threading.RLock()
        self._update_callbacks: List[Callable] = []
        
        # Load initial configuration
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load and merge configuration files."""
        with self._lock:
            # Load base configuration
            self._base_config = self._load_config_file(self.config_path)
            
            # Load environment-specific configuration
            env_config_path = self._get_env_config_path()
            if env_config_path.exists():
                self._env_config = self._load_config_file(env_config_path)
            else:
                self._env_config = {}
            
            # Merge configurations
            self._merged_config = self._merge_configs(self._base_config, self._env_config)
            
            # Validate configuration
            if self.schema:
                errors = self.schema.validate(self._merged_config)
                if errors:
                    raise ConfigurationValidationError(f"Configuration validation failed: {errors}")
            
            # Store in history
            self._config_history.append({
                'timestamp': datetime.now(),
                'config': self._merged_config.copy(),
                'environment': self.environment
            })
            
            if self.logger:
                self.logger.info(f"Configuration loaded for environment: {self.environment}")
    
    def _load_config_file(self, filepath: Path) -> Dict[str, Any]:
        """Load configuration from a file."""
        if not filepath.exists():
            raise ConfigurationError(f"Configuration file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif filepath.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    raise ConfigurationError(f"Unsupported configuration file format: {filepath.suffix}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration file {filepath}: {e}") from e
    
    def _get_env_config_path(self) -> Path:
        """Get environment-specific configuration file path."""
        base_name = self.config_path.stem
        suffix = self.config_path.suffix
        return self.config_path.parent / f"{base_name}.{self.environment}{suffix}"
    
    def _merge_configs(self, base: Dict[str, Any], env: Dict[str, Any]) -> Dict[str, Any]:
        """Merge base and environment configurations."""
        merged = base.copy()
        
        def deep_merge(base_dict: Dict[str, Any], env_dict: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in env_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    base_dict[key] = deep_merge(base_dict[key], value)
                else:
                    base_dict[key] = value
            return base_dict
        
        return deep_merge(merged, env)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        keys = key.split('.')
        value = self._merged_config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value using dot notation."""
        with self._lock:
            keys = key.split('.')
            config = self._merged_config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
            
            # Notify callbacks
            self._notify_update_callbacks(key, value)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a configuration section."""
        return self.get(section, {})
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> None:
        """Update a configuration section."""
        with self._lock:
            if section not in self._merged_config:
                self._merged_config[section] = {}
            
            self._merged_config[section].update(updates)
            
            # Notify callbacks
            for key, value in updates.items():
                self._notify_update_callbacks(f"{section}.{key}", value)
    
    def save_config(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file."""
        filepath = Path(filepath) if filepath else self.config_path
        
        with self._lock:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    if filepath.suffix.lower() in ['.yaml', '.yml']:
                        yaml.dump(self._merged_config, f, default_flow_style=False, indent=2)
                    elif filepath.suffix.lower() == '.json':
                        json.dump(self._merged_config, f, indent=2)
                    else:
                        raise ConfigurationError(f"Unsupported file format: {filepath.suffix}")
                
                if self.logger:
                    self.logger.info(f"Configuration saved to: {filepath}")
                    
            except Exception as e:
                raise ConfigurationError(f"Error saving configuration: {e}") from e
    
    def reload_config(self) -> None:
        """Reload configuration from files."""
        with self._lock:
            self._load_configuration()
            
            # Notify all callbacks
            for callback in self._update_callbacks:
                try:
                    callback(self._merged_config)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error in config update callback: {e}")
    
    def add_update_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback for configuration updates."""
        self._update_callbacks.append(callback)
    
    def remove_update_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a configuration update callback."""
        if callback in self._update_callbacks:
            self._update_callbacks.remove(callback)
    
    def start_watching(self, interval: float = 1.0) -> None:
        """Start watching configuration files for changes."""
        if self._watcher is None:
            self._watcher = ConfigWatcher(self._on_config_change, self.logger)
        
        self._watcher.watch_file(self.config_path)
        env_config_path = self._get_env_config_path()
        if env_config_path.exists():
            self._watcher.watch_file(env_config_path)
        
        self._watcher.start_watching(interval)
    
    def stop_watching(self) -> None:
        """Stop watching configuration files."""
        if self._watcher:
            self._watcher.stop_watching()
    
    def _on_config_change(self, filepath: str) -> None:
        """Handle configuration file changes."""
        if self.logger:
            self.logger.info(f"Configuration file changed, reloading: {filepath}")
        
        self.reload_config()
    
    def get_config_hash(self) -> str:
        """Get a hash of the current configuration."""
        config_str = json.dumps(self._merged_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_config_history(self) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        return self._config_history.copy()
    
    def store_secret(self, key: str, value: str) -> None:
        """Store an encrypted secret."""
        self._secret_manager.store_secret(key, value)
    
    def get_secret(self, key: str) -> Optional[str]:
        """Get a decrypted secret."""
        return self._secret_manager.get_secret(key)
    
    def validate_config(self) -> Dict[str, List[str]]:
        """Validate current configuration against schema."""
        if not self.schema:
            return {}
        
        return self.schema.validate(self._merged_config)
    
    def export_config(self, filepath: Union[str, Path], format: str = "yaml") -> None:
        """Export configuration to a file."""
        filepath = Path(filepath)
        
        with self._lock:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    if format.lower() == 'yaml':
                        yaml.dump(self._merged_config, f, default_flow_style=False, indent=2)
                    elif format.lower() == 'json':
                        json.dump(self._merged_config, f, indent=2)
                    else:
                        raise ConfigurationError(f"Unsupported export format: {format}")
                
                if self.logger:
                    self.logger.info(f"Configuration exported to: {filepath}")
                    
            except Exception as e:
                raise ConfigurationError(f"Error exporting configuration: {e}") from e
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables that override configuration."""
        env_vars = {}
        for key, value in os.environ.items():
            if key.startswith('ML_CONFIG_'):
                config_key = key[10:].lower().replace('_', '.')
                env_vars[config_key] = value
        return env_vars
    
    def apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        env_vars = self.get_environment_variables()
        
        with self._lock:
            for key, value in env_vars.items():
                # Try to convert to appropriate type
                try:
                    # Try JSON parsing first
                    parsed_value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    # Fall back to string
                    parsed_value = value
                
                self.set(key, parsed_value)
            
            if env_vars and self.logger:
                self.logger.info(f"Applied {len(env_vars)} environment variable overrides")
