#!/usr/bin/env python3
"""
Test script for ML1 loan prediction system
Validates that all modules can be imported and basic functionality works
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test core modules
        from src.core import ConfigManager, Logger
        print("✓ Core modules imported successfully")
        
        # Test data modules
        from src.data import LoanDataProcessor, FeatureEngineer, DataValidator
        print("✓ Data modules imported successfully")
        
        # Test model modules
        from src.models import (
            BaseMLModel, LogisticRegressionModel, RandomForestModel,
            XGBoostModel, LightGBMModel, SVMModel, NaiveBayesModel, KNNModel,
            VotingEnsembleModel, StackingEnsembleModel, BaggingEnsembleModel
        )
        print("✓ Model modules imported successfully")
        
        # Test training modules
        from src.training import ModelTrainer, HyperparameterOptimizer
        print("✓ Training modules imported successfully")
        
        # Test evaluation modules
        from src.evaluation import ModelEvaluator, ResultVisualizer
        print("✓ Evaluation modules imported successfully")
        
        # Test utils modules
        from src.utils import setup_directories, save_results, load_results
        from src.utils import plot_feature_importance, plot_training_history
        print("✓ Utils modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        from src.core import ConfigManager
        
        # Test loading config
        config_manager = ConfigManager("config.yaml")
        print("✓ Configuration loaded successfully")
        
        # Test getting config values
        model_config = config_manager.get_model_config()
        data_config = config_manager.get_data_config()
        train_config = config_manager.get_train_config()
        
        print(f"✓ Model config: {model_config.get('model_name', 'N/A')}")
        print(f"✓ Data config keys: {list(data_config.keys())}")
        print(f"✓ Train config keys: {list(train_config.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_data_processing():
    """Test data processing functionality"""
    print("\nTesting data processing...")
    
    try:
        from src.core import ConfigManager, Logger
        from src.data import LoanDataProcessor, DataValidator
        
        # Initialize components
        config_manager = ConfigManager("config.yaml")
        logger = Logger(config_manager.get_logger_config())
        processor = LoanDataProcessor(config_manager.get_data_config(), logger)
        validator = DataValidator(config_manager.get_data_config(), logger)
        
        # Test data loading
        data_path = "data/train_u6lujuX_CVtuZ9i.csv"
        if os.path.exists(data_path):
            df = processor.load_data(data_path)
            print(f"✓ Data loaded successfully: {df.shape}")
            
            # Test data validation
            validation_results = validator.validate_data_quality(df)
            print(f"✓ Data validation completed: {validation_results['quality_score']:.2f}")
            
            # Test data preprocessing
            X, y = processor.preprocess(df)
            print(f"✓ Data preprocessing completed: X={X.shape}, y={y.shape}")
            
            return True
        else:
            print(f"✗ Data file not found: {data_path}")
            return False
            
    except Exception as e:
        print(f"✗ Data processing error: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    
    try:
        from src.core import ConfigManager, Logger
        from src.models import RandomForestModel, LogisticRegressionModel
        
        # Initialize components
        config_manager = ConfigManager("config.yaml")
        logger = Logger(config_manager.get_logger_config())
        
        # Test RandomForest model
        rf_config = {
            'model_name': 'RandomForest',
            'model_type': 'classification',
            'model_params': {'n_estimators': 10, 'random_state': 42}
        }
        rf_model = RandomForestModel(rf_config, logger)
        rf_model.model = rf_model.build_model()
        print("✓ RandomForest model created successfully")
        
        # Test LogisticRegression model
        lr_config = {
            'model_name': 'LogisticRegression',
            'model_type': 'classification',
            'model_params': {'random_state': 42, 'max_iter': 100}
        }
        lr_model = LogisticRegressionModel(lr_config, logger)
        lr_model.model = lr_model.build_model()
        print("✓ LogisticRegression model created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def test_training():
    """Test training functionality"""
    print("\nTesting training functionality...")
    
    try:
        from src.core import ConfigManager, Logger
        from src.models import RandomForestModel
        from src.training import ModelTrainer
        import numpy as np
        
        # Initialize components
        config_manager = ConfigManager("config.yaml")
        logger = Logger(config_manager.get_logger_config())
        
        # Create dummy data
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # Create model
        model_config = {
            'model_name': 'RandomForest',
            'model_type': 'classification',
            'model_params': {'n_estimators': 10, 'random_state': 42}
        }
        model = RandomForestModel(model_config, logger)
        
        # Create trainer
        trainer = ModelTrainer(config_manager.get_train_config(), logger)
        trainer.set_model(model)
        
        # Test training
        training_history = trainer.train(X, y)
        print("✓ Training completed successfully")
        
        # Test evaluation
        metrics = trainer.evaluate(X, y)
        print(f"✓ Evaluation completed: accuracy={metrics.get('accuracy', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training error: {e}")
        return False

def test_evaluation():
    """Test evaluation functionality"""
    print("\nTesting evaluation functionality...")
    
    try:
        from src.core import ConfigManager, Logger
        from src.evaluation import ModelEvaluator
        import numpy as np
        
        # Initialize components
        config_manager = ConfigManager("config.yaml")
        logger = Logger(config_manager.get_logger_config())
        evaluator = ModelEvaluator(config_manager.get_metrics_config(), logger)
        
        # Create dummy data
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        y_proba = np.random.rand(100, 2)
        
        # Test metrics computation
        metrics = evaluator.compute_metrics(y_true, y_pred, y_proba)
        print(f"✓ Metrics computed: {list(metrics.keys())}")
        
        # Test report generation
        report = evaluator.generate_evaluation_report(y_true, y_pred, y_proba, "TestModel")
        print("✓ Evaluation report generated")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation error: {e}")
        return False

def main():
    """Run all tests"""
    print("ML1 System Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_loading,
        test_data_processing,
        test_model_creation,
        test_training,
        test_evaluation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
