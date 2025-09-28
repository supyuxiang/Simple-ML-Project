#!/usr/bin/env python3
"""
Production System Test Suite

This module provides comprehensive testing for the production-grade ML pipeline
with all advanced features integrated.
"""

import asyncio
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from src.core import (
    AdvancedConfigManager, Logger, get_component, list_components,
    start_monitoring, stop_monitoring, get_monitoring_system,
    get_cache_manager, set_cache_backend, DiskCacheBackend
)
from src.core.exceptions import MLProjectError


class ProductionSystemTester:
    """Production system tester."""
    
    def __init__(self):
        self.test_results = []
        self.logger = None
    
    def run_test(self, test_name: str, test_func):
        """Run a single test."""
        try:
            print(f"Testing {test_name}...")
            result = test_func()
            if result:
                print(f"✓ {test_name} passed")
                self.test_results.append((test_name, True, None))
            else:
                print(f"✗ {test_name} failed")
                self.test_results.append((test_name, False, "Test returned False"))
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            self.test_results.append((test_name, False, str(e)))
    
    def test_imports(self):
        """Test all module imports."""
        try:
            from src.core import (
                AdvancedConfigManager, ConfigSchema, Logger,
                ComponentRegistry, CacheManager, MonitoringSystem,
                TaskScheduler, profile_function, add_metric
            )
            from src.training.trainer import AdvancedModelTrainer
            from src.data.processor import AdvancedDataProcessor
            from src.models import XGBoostModel, RandomForestModel
            from src.evaluation.evaluator import ModelEvaluator
            return True
        except ImportError as e:
            print(f"Import error: {e}")
            return False
    
    def test_registry_system(self):
        """Test component registry system."""
        try:
            # Test component listing
            components = list_components()
            assert isinstance(components, dict)
            
            # Test component retrieval
            if "models" in components and "xgboost" in components["models"]:
                model = get_component("xgboost", "models", {}, None)
                assert model is not None
            
            return True
        except Exception as e:
            print(f"Registry test error: {e}")
            return False
    
    def test_configuration_management(self):
        """Test advanced configuration management."""
        try:
            config_manager = AdvancedConfigManager("config.yaml")
            
            # Test configuration access
            model_config = config_manager.get_section("Model")
            assert isinstance(model_config, dict)
            
            # Test configuration validation
            validation_errors = config_manager.validate_config()
            assert isinstance(validation_errors, dict)
            
            return True
        except Exception as e:
            print(f"Configuration test error: {e}")
            return False
    
    def test_caching_system(self):
        """Test caching system."""
        try:
            cache_manager = get_cache_manager()
            
            # Test basic cache operations
            cache_manager.set("test_key", "test_value")
            value = cache_manager.get("test_key")
            assert value == "test_value"
            
            # Test cache statistics
            stats = cache_manager.get_stats()
            assert stats.hits >= 0
            
            return True
        except Exception as e:
            print(f"Caching test error: {e}")
            return False
    
    def test_monitoring_system(self):
        """Test monitoring system."""
        try:
            monitoring = get_monitoring_system()
            
            # Test metric collection
            add_metric("test.metric", 42.5)
            metrics = monitoring.collect_metrics()
            assert len(metrics) > 0
            
            # Test health checks
            health_checks = monitoring.perform_health_checks()
            assert len(health_checks) > 0
            
            return True
        except Exception as e:
            print(f"Monitoring test error: {e}")
            return False
    
    def test_async_tasks(self):
        """Test async task system."""
        try:
            from src.core.async_tasks import submit_function, get_scheduler
            
            # Test function submission
            def test_func(x):
                return x * 2
            
            task_id = submit_function(test_func, args=(5,))
            assert task_id is not None
            
            return True
        except Exception as e:
            print(f"Async tasks test error: {e}")
            return False
    
    def test_data_processing(self):
        """Test advanced data processing."""
        try:
            from src.data.processor import AdvancedDataProcessor
            from src.core import AdvancedConfigManager, Logger
            
            # Initialize components
            config_manager = AdvancedConfigManager("config.yaml")
            logger = Logger(config_manager.get_section("Logger"))
            processor = AdvancedDataProcessor(config_manager, logger)
            
            # Test data loading (if data file exists)
            data_path = "data/train_u6lujuX_CVtuZ9i.csv"
            if Path(data_path).exists():
                df = processor.load_data(data_path)
                assert len(df) > 0
                assert len(df.columns) > 0
            
            return True
        except Exception as e:
            print(f"Data processing test error: {e}")
            return False
    
    def test_model_training(self):
        """Test advanced model training."""
        try:
            from src.training.trainer import AdvancedModelTrainer
            from src.core import AdvancedConfigManager, Logger
            import numpy as np
            
            # Initialize components
            config_manager = AdvancedConfigManager("config.yaml")
            logger = Logger(config_manager.get_section("Logger"))
            trainer = AdvancedModelTrainer(config_manager, logger)
            
            # Create dummy data for testing
            X = np.random.rand(100, 10)
            y = np.random.randint(0, 2, 100)
            
            # Test training (without actual model)
            # This will test the training infrastructure
            assert trainer.cv_folds > 0
            assert trainer.random_seed is not None
            
            return True
        except Exception as e:
            print(f"Model training test error: {e}")
            return False
    
    def test_pipeline_integration(self):
        """Test complete pipeline integration."""
        try:
            from src.core import AdvancedConfigManager, Logger
            from src.training.trainer import AdvancedModelTrainer
            from src.data.processor import AdvancedDataProcessor
            from src.evaluation.evaluator import ModelEvaluator
            
            # Initialize all components
            config_manager = AdvancedConfigManager("config.yaml")
            logger = Logger(config_manager.get_section("Logger"))
            
            # Test component initialization
            processor = AdvancedDataProcessor(config_manager, logger)
            trainer = AdvancedModelTrainer(config_manager, logger)
            evaluator = ModelEvaluator(config_manager.get_section("Metrics"), logger)
            
            # Verify all components are initialized
            assert processor is not None
            assert trainer is not None
            assert evaluator is not None
            
            return True
        except Exception as e:
            print(f"Pipeline integration test error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests."""
        print("Production System Test Suite")
        print("=" * 50)
        
        # Run all tests
        self.run_test("Module Imports", self.test_imports)
        self.run_test("Registry System", self.test_registry_system)
        self.run_test("Configuration Management", self.test_configuration_management)
        self.run_test("Caching System", self.test_caching_system)
        self.run_test("Monitoring System", self.test_monitoring_system)
        self.run_test("Async Tasks", self.test_async_tasks)
        self.run_test("Data Processing", self.test_data_processing)
        self.run_test("Model Training", self.test_model_training)
        self.run_test("Pipeline Integration", self.test_pipeline_integration)
        
        # Print results
        print("\n" + "=" * 50)
        print("Test Results Summary")
        print("=" * 50)
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        for test_name, success, error in self.test_results:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{status} {test_name}")
            if not success and error:
                print(f"    Error: {error}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Production system is ready.")
            return True
        else:
            print("❌ Some tests failed. Please check the errors above.")
            return False


def main():
    """Main test function."""
    tester = ProductionSystemTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
