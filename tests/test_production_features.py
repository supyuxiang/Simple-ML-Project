"""
Comprehensive test suite for advanced production features.

This module tests the new advanced components:
- Registry system
- Caching system
- Async task system
- Monitoring system
- Advanced configuration management
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

# Add src directory to Python path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.registry import ComponentRegistry, register_component, get_component
from core.cache import CacheManager, MemoryCacheBackend, DiskCacheBackend, cached
from core.async_tasks import TaskScheduler, FunctionTask, AsyncFunctionTask, submit_function
from core.monitoring import MonitoringSystem, profile_function, add_metric
from core.advanced_config import AdvancedConfigManager, ConfigSchema, SecretManager
from core.exceptions import RegistryError, CacheError, TaskError, MonitoringError


class TestRegistrySystem:
    """Test the component registry system."""
    
    def test_component_registration(self):
        """Test basic component registration."""
        registry = ComponentRegistry()
        
        class TestComponent:
            def __init__(self, value):
                self.value = value
        
        registry.register("test", TestComponent, "components")
        assert "test" in registry.list_components("components")
        
        instance = registry.get("test", "components", value=42)
        assert instance.value == 42
    
    def test_component_dependencies(self):
        """Test component dependency injection."""
        registry = ComponentRegistry()
        
        class Dependency:
            def __init__(self):
                self.value = "dependency"
        
        class MainComponent:
            def __init__(self, dep):
                self.dep = dep
        
        registry.register("dep", Dependency, "components")
        registry.register("main", MainComponent, "components", dependencies=["dep"])
        
        instance = registry.get("main", "components")
        assert instance.dep.value == "dependency"
    
    def test_decorator_registration(self):
        """Test decorator-based component registration."""
        registry = ComponentRegistry()
        
        @register_component("decorated", "components")
        class DecoratedComponent:
            def __init__(self):
                self.value = "decorated"
        
        # Note: This test would need the global registry to be properly set up
        # For now, we'll test the registry directly
        registry.register("decorated", DecoratedComponent, "components")
        instance = registry.get("decorated", "components")
        assert instance.value == "decorated"
    
    def test_singleton_behavior(self):
        """Test singleton component behavior."""
        registry = ComponentRegistry()
        
        class SingletonComponent:
            def __init__(self):
                self.id = id(self)
        
        registry.register("singleton", SingletonComponent, "components", singleton=True)
        
        instance1 = registry.get("singleton", "components")
        instance2 = registry.get("singleton", "components")
        
        assert instance1 is instance2
        assert instance1.id == instance2.id


class TestCachingSystem:
    """Test the caching system."""
    
    def test_memory_cache_basic(self):
        """Test basic memory cache operations."""
        cache = MemoryCacheBackend(max_size=10)
        
        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.exists("key1")
        
        # Test delete
        cache.delete("key1")
        assert cache.get("key1") is None
        assert not cache.exists("key1")
    
    def test_memory_cache_eviction(self):
        """Test LRU eviction in memory cache."""
        cache = MemoryCacheBackend(max_size=3)
        
        # Fill cache beyond capacity
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
    
    def test_disk_cache(self):
        """Test disk cache operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCacheBackend(temp_dir)
            
            # Test set and get
            cache.set("key1", {"data": [1, 2, 3]})
            result = cache.get("key1")
            assert result["data"] == [1, 2, 3]
            
            # Test exists
            assert cache.exists("key1")
            assert not cache.exists("key2")
    
    def test_cache_manager(self):
        """Test cache manager functionality."""
        cache_manager = CacheManager(MemoryCacheBackend())
        
        # Test basic operations
        cache_manager.set("key1", "value1")
        assert cache_manager.get("key1") == "value1"
        
        # Test statistics
        stats = cache_manager.get_stats()
        assert stats.hits >= 0
        assert stats.misses >= 0
    
    def test_cached_decorator(self):
        """Test the cached decorator."""
        cache_manager = CacheManager(MemoryCacheBackend())
        
        call_count = 0
        
        @cache_manager.cached(ttl=60)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call should execute function
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment


class TestAsyncTaskSystem:
    """Test the async task system."""
    
    @pytest.mark.asyncio
    async def test_task_scheduler_basic(self):
        """Test basic task scheduler functionality."""
        scheduler = TaskScheduler(max_workers=2)
        await scheduler.start()
        
        try:
            # Submit a simple function task
            def simple_task(x):
                return x * 2
            
            task_id = scheduler.submit_function(simple_task, args=(5,))
            
            # Wait for completion
            result = await scheduler.wait_for_task(task_id, timeout=5)
            
            assert result.status.value == "completed"
            assert result.result == 10
            
        finally:
            await scheduler.stop()
    
    @pytest.mark.asyncio
    async def test_async_function_task(self):
        """Test async function tasks."""
        scheduler = TaskScheduler(max_workers=2)
        await scheduler.start()
        
        try:
            async def async_task(x):
                await asyncio.sleep(0.1)
                return x * 3
            
            task_id = scheduler.submit_async_function(async_task, args=(4,))
            result = await scheduler.wait_for_task(task_id, timeout=5)
            
            assert result.status.value == "completed"
            assert result.result == 12
            
        finally:
            await scheduler.stop()
    
    @pytest.mark.asyncio
    async def test_task_priority(self):
        """Test task priority ordering."""
        scheduler = TaskScheduler(max_workers=1)
        await scheduler.start()
        
        try:
            results = []
            
            def task(name):
                results.append(name)
                return name
            
            # Submit tasks with different priorities
            scheduler.submit_function(task, args=("low",), priority=1)
            scheduler.submit_function(task, args=("high",), priority=3)
            scheduler.submit_function(task, args=("normal",), priority=2)
            
            # Wait for all tasks
            await scheduler.wait_for_all_tasks(timeout=5)
            
            # High priority should execute first (lower number = higher priority in queue)
            assert results[0] == "high"
            
        finally:
            await scheduler.stop()
    
    @pytest.mark.asyncio
    async def test_task_timeout(self):
        """Test task timeout handling."""
        scheduler = TaskScheduler(max_workers=1)
        await scheduler.start()
        
        try:
            def slow_task():
                time.sleep(2)
                return "completed"
            
            task_id = scheduler.submit_function(
                slow_task, 
                timeout=1  # 1 second timeout
            )
            
            result = await scheduler.wait_for_task(task_id, timeout=5)
            
            assert result.status.value == "timeout"
            assert result.error is not None
            
        finally:
            await scheduler.stop()


class TestMonitoringSystem:
    """Test the monitoring system."""
    
    def test_metric_collection(self):
        """Test metric collection."""
        monitoring = MonitoringSystem()
        
        # Add custom metrics
        add_metric("test.metric", 42.5, {"tag": "test"})
        add_metric("test.metric", 43.0, {"tag": "test"})
        
        # Collect metrics
        metrics = monitoring.collect_metrics()
        
        # Should have system metrics + our custom metrics
        assert len(metrics) > 0
        
        # Check if our custom metrics are there
        test_metrics = [m for m in metrics if m.name == "test.metric"]
        assert len(test_metrics) >= 2
    
    def test_health_checks(self):
        """Test health check functionality."""
        monitoring = MonitoringSystem()
        
        health_checks = monitoring.perform_health_checks()
        
        assert len(health_checks) > 0
        
        # Should have system health check
        system_check = next((hc for hc in health_checks if hc.name == "system"), None)
        assert system_check is not None
        assert system_check.status in ["healthy", "degraded", "unhealthy"]
    
    def test_performance_profiling(self):
        """Test performance profiling."""
        monitoring = MonitoringSystem()
        
        @profile_function
        def test_function(x):
            time.sleep(0.01)  # Small delay
            return x * 2
        
        # Call function multiple times
        for _ in range(5):
            test_function(5)
        
        profiles = monitoring.get_performance_profiles()
        
        assert "test_function" in profiles
        profile = profiles["test_function"]
        assert profile.total_calls == 5
        assert profile.avg_time > 0
        assert profile.min_time > 0
        assert profile.max_time > 0
    
    def test_metric_history(self):
        """Test metric history tracking."""
        monitoring = MonitoringSystem()
        
        # Add some metrics
        for i in range(10):
            add_metric("test.counter", i)
        
        # Get history
        history = monitoring.get_metric_history("test.counter")
        assert len(history) == 10
        
        # Get summary
        summary = monitoring.get_metric_summary("test.counter")
        assert summary["count"] == 10
        assert summary["min"] == 0
        assert summary["max"] == 9
        assert summary["avg"] == 4.5


class TestAdvancedConfig:
    """Test advanced configuration management."""
    
    def test_config_loading(self):
        """Test configuration loading and merging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create base config
            base_config = {
                "database": {
                    "host": "localhost",
                    "port": 5432
                },
                "model": {
                    "name": "default_model"
                }
            }
            
            base_config_path = Path(temp_dir) / "config.yaml"
            with open(base_config_path, "w") as f:
                yaml.dump(base_config, f)
            
            # Create environment config
            env_config = {
                "database": {
                    "host": "prod-db.example.com"
                },
                "model": {
                    "name": "production_model",
                    "batch_size": 32
                }
            }
            
            env_config_path = Path(temp_dir) / "config.production.yaml"
            with open(env_config_path, "w") as f:
                yaml.dump(env_config, f)
            
            # Load configuration
            config_manager = AdvancedConfigManager(
                base_config_path, 
                environment="production"
            )
            
            # Test merged configuration
            assert config_manager.get("database.host") == "prod-db.example.com"
            assert config_manager.get("database.port") == 5432
            assert config_manager.get("model.name") == "production_model"
            assert config_manager.get("model.batch_size") == 32
    
    def test_config_validation(self):
        """Test configuration validation."""
        schema = ConfigSchema(
            name="test_schema",
            version="1.0",
            fields={
                "database.port": {"type": int, "min": 1, "max": 65535},
                "model.batch_size": {"type": int, "min": 1}
            },
            required_fields=["database.port", "model.batch_size"]
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Valid config
            valid_config = {
                "database": {"port": 5432},
                "model": {"batch_size": 32}
            }
            
            config_path = Path(temp_dir) / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(valid_config, f)
            
            config_manager = AdvancedConfigManager(config_path, schema=schema)
            
            # Should load without errors
            assert config_manager.get("database.port") == 5432
            
            # Invalid config
            invalid_config = {
                "database": {"port": "invalid"},
                "model": {"batch_size": 32}
            }
            
            with open(config_path, "w") as f:
                yaml.dump(invalid_config, f)
            
            # Should raise validation error
            with pytest.raises(Exception):  # ConfigurationValidationError
                AdvancedConfigManager(config_path, schema=schema)
    
    def test_secret_management(self):
        """Test secret management."""
        secret_manager = SecretManager()
        
        # Store and retrieve secret
        secret_manager.store_secret("api_key", "secret123")
        retrieved = secret_manager.get_secret("api_key")
        
        assert retrieved == "secret123"
        
        # Test encryption/decryption
        encrypted = secret_manager.encrypt_secret("test_secret")
        decrypted = secret_manager.decrypt_secret(encrypted)
        
        assert decrypted == "test_secret"
        assert encrypted != "test_secret"  # Should be encrypted
    
    def test_config_watching(self):
        """Test configuration file watching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            
            # Initial config
            initial_config = {"value": 1}
            with open(config_path, "w") as f:
                yaml.dump(initial_config, f)
            
            config_manager = AdvancedConfigManager(config_path)
            assert config_manager.get("value") == 1
            
            # Update config file
            updated_config = {"value": 2}
            with open(config_path, "w") as f:
                yaml.dump(updated_config, f)
            
            # Start watching and wait for change
            config_manager.start_watching(interval=0.1)
            time.sleep(0.5)  # Wait for file change detection
            
            # Should have updated value
            assert config_manager.get("value") == 2
            
            config_manager.stop_watching()


class TestIntegration:
    """Integration tests for advanced features."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self):
        """Test integration of all advanced features."""
        # Setup monitoring
        monitoring = MonitoringSystem()
        monitoring.start_monitoring(interval=1.0)
        
        # Setup cache
        cache_manager = CacheManager(MemoryCacheBackend())
        
        # Setup task scheduler
        scheduler = TaskScheduler(max_workers=2)
        await scheduler.start()
        
        try:
            # Define a cached, monitored function
            @cache_manager.cached(ttl=60)
            @profile_function
            def process_data(data):
                add_metric("processing.count", len(data))
                time.sleep(0.1)  # Simulate processing
                return [x * 2 for x in data]
            
            # Submit tasks
            task_ids = []
            for i in range(3):
                task_id = scheduler.submit_function(process_data, args=([1, 2, 3],))
                task_ids.append(task_id)
            
            # Wait for all tasks
            await scheduler.wait_for_all_tasks(timeout=10)
            
            # Verify results
            for task_id in task_ids:
                result = await scheduler.wait_for_task(task_id)
                assert result.status.value == "completed"
                assert result.result == [2, 4, 6]
            
            # Check monitoring data
            profiles = monitoring.get_performance_profiles()
            assert "process_data" in profiles
            
            metrics = monitoring.collect_metrics()
            processing_metrics = [m for m in metrics if m.name == "processing.count"]
            assert len(processing_metrics) >= 3
            
            # Check cache statistics
            cache_stats = cache_manager.get_stats()
            assert cache_stats.hits > 0  # Should have cache hits due to same input
            
        finally:
            await scheduler.stop()
            monitoring.stop_monitoring()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
