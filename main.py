#!/usr/bin/env python3
"""
Main entry point for ML1 loan prediction project
Implements a complete machine learning pipeline with enterprise-grade architecture
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.core import ConfigManager, Logger
from src.data import LoanDataProcessor, FeatureEngineer, DataValidator
from src.models import (
    LogisticRegressionModel, RandomForestModel, XGBoostModel, 
    LightGBMModel, SVMModel, NaiveBayesModel, KNNModel
)
from src.training import ModelTrainer
from src.evaluation import ModelEvaluator
from src.utils import setup_directories, save_results


class ML1Pipeline:
    """
    Main pipeline class for ML1 loan prediction project
    Orchestrates the entire machine learning workflow
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the ML1 pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path)
        self.logger = None
        self.data_processor = None
        self.feature_engineer = None
        self.data_validator = None
        self.models = {}
        self.trainer = None
        self.evaluator = None
        
        # Setup directories
        self._setup_directories()
        
        # Initialize logger
        self._initialize_logger()
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("ML1 Pipeline initialized successfully")
    
    def _setup_directories(self) -> None:
        """
        Setup required directories
        """
        base_path = Path(__file__).parent
        self.config_manager.update_paths(str(base_path))
        
        directories = [
            'outputs/models',
            'outputs/curves', 
            'outputs/reports',
            'logs',
            'checkpoints'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _initialize_logger(self) -> None:
        """
        Initialize the logger
        """
        logger_config = self.config_manager.get_logger_config()
        self.logger = Logger(logger_config, log_dir="logs")
    
    def _initialize_components(self) -> None:
        """
        Initialize all pipeline components
        """
        # Data processing components
        data_config = self.config_manager.get_data_config()
        self.data_processor = LoanDataProcessor(data_config, self.logger)
        self.feature_engineer = FeatureEngineer(data_config, self.logger)
        self.data_validator = DataValidator(data_config, self.logger)
        
        # Training and evaluation components
        train_config = self.config_manager.get_train_config()
        metrics_config = self.config_manager.get_metrics_config()
        
        self.trainer = ModelTrainer(train_config, self.logger)
        self.evaluator = ModelEvaluator(metrics_config, self.logger)
        
        self.logger.info("All components initialized")
    
    def load_and_validate_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and validate data
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Loaded and validated DataFrame
        """
        self.logger.info("Loading and validating data...")
        
        # Load data
        df = self.data_processor.load_data(data_path)
        
        # Validate data quality
        validation_results = self.data_validator.validate_data_quality(df)
        
        # Log validation results
        self.logger.log_data_info(validation_results['basic_info'])
        
        # Generate validation report
        validation_report = self.data_validator.generate_validation_report(validation_results)
        self.logger.info(f"Data validation report:\n{validation_report}")
        
        # Get recommendations
        recommendations = self.data_validator.get_recommendations(validation_results)
        if recommendations:
            self.logger.warning("Data quality recommendations:")
            for rec in recommendations:
                self.logger.warning(f"  - {rec}")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> tuple:
        """
        Preprocess the data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val, X_test, y_test)
        """
        self.logger.info("Preprocessing data...")
        
        # Preprocess data
        X, y = self.data_processor.preprocess(df)
        
        # Feature engineering
        X_engineered, feature_names = self.feature_engineer.engineer_features(X, y, self.data_processor.feature_names)
        
        # Split data
        X_train, X_val, y_train, y_val = self.data_processor.split_data(X_engineered, y)
        
        # For this example, we'll use validation set as test set
        # In practice, you might want to split further
        X_test, y_test = X_val, y_val
        
        self.logger.info(f"Data preprocessing completed:")
        self.logger.info(f"  Train set: {X_train.shape}")
        self.logger.info(f"  Validation set: {X_val.shape}")
        self.logger.info(f"  Test set: {X_test.shape}")
        
        return X_train, X_val, y_train, y_val, X_test, y_test
    
    def initialize_models(self) -> None:
        """
        Initialize all models to be trained
        """
        self.logger.info("Initializing models...")
        
        model_config = self.config_manager.get_model_config()
        model_name = model_config.get('model_name', 'RandomForest')
        
        # Model configurations
        model_configs = {
            'LogisticRegression': {
                'model_name': 'LogisticRegression',
                'model_type': 'classification',
                'model_params': {
                    'random_state': 42,
                    'max_iter': 1000,
                    'solver': 'liblinear'
                }
            },
            'RandomForest': {
                'model_name': 'RandomForest',
                'model_type': 'classification',
                'model_params': {
                    'n_estimators': 100,
                    'random_state': 42,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2
                }
            },
            'XGBoost': {
                'model_name': 'XGBoost',
                'model_type': 'classification',
                'model_params': {
                    'n_estimators': 100,
                    'random_state': 42,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                }
            },
            'LightGBM': {
                'model_name': 'LightGBM',
                'model_type': 'classification',
                'model_params': {
                    'n_estimators': 100,
                    'random_state': 42,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'verbose': -1
                }
            },
            'SVM': {
                'model_name': 'SVM',
                'model_type': 'classification',
                'model_params': {
                    'random_state': 42,
                    'probability': True,
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale'
                }
            },
            'NaiveBayes': {
                'model_name': 'NaiveBayes',
                'model_type': 'classification',
                'model_params': {}
            },
            'KNN': {
                'model_name': 'KNN',
                'model_type': 'classification',
                'model_params': {
                    'n_neighbors': 5,
                    'weights': 'uniform',
                    'algorithm': 'auto'
                }
            }
        }
        
        # Initialize models
        models_to_train = [model_name] if model_name != 'all' else list(model_configs.keys())
        
        for model_name in models_to_train:
            if model_name in model_configs:
                try:
                    if model_name == 'LogisticRegression':
                        self.models[model_name] = LogisticRegressionModel(model_configs[model_name], self.logger)
                    elif model_name == 'RandomForest':
                        self.models[model_name] = RandomForestModel(model_configs[model_name], self.logger)
                    elif model_name == 'XGBoost':
                        self.models[model_name] = XGBoostModel(model_configs[model_name], self.logger)
                    elif model_name == 'LightGBM':
                        self.models[model_name] = LightGBMModel(model_configs[model_name], self.logger)
                    elif model_name == 'SVM':
                        self.models[model_name] = SVMModel(model_configs[model_name], self.logger)
                    elif model_name == 'NaiveBayes':
                        self.models[model_name] = NaiveBayesModel(model_configs[model_name], self.logger)
                    elif model_name == 'KNN':
                        self.models[model_name] = KNNModel(model_configs[model_name], self.logger)
                    
                    self.logger.info(f"Initialized {model_name} model")
                    
                except ImportError as e:
                    self.logger.warning(f"Could not initialize {model_name}: {e}")
                except Exception as e:
                    self.logger.error(f"Error initializing {model_name}: {e}")
        
        self.logger.info(f"Initialized {len(self.models)} models")
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Train all models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary containing training results
        """
        self.logger.info("Training models...")
        
        training_results = {}
        
        for model_name, model in self.models.items():
            try:
                self.logger.info(f"Training {model_name}...")
                
                # Train model
                model.fit(X_train, y_train, X_val, y_val)
                
                # Evaluate on validation set
                val_predictions = model.predict(X_val)
                val_probabilities = model.predict_proba(X_val)
                
                # Calculate metrics
                val_metrics = self.evaluator.compute_metrics(y_val, val_predictions, val_probabilities)
                model.update_metrics(val_metrics, 'val')
                
                # Store results
                training_results[model_name] = {
                    'model': model,
                    'val_metrics': val_metrics,
                    'feature_importance': model.get_feature_importance()
                }
                
                self.logger.log_training_progress(
                    epoch=1,  # Single epoch for sklearn models
                    train_loss=0.0,  # Not applicable for sklearn
                    val_loss=1 - val_metrics.get('accuracy', 0),  # Use 1-accuracy as proxy
                    val_metrics=val_metrics
                )
                
                self.logger.info(f"{model_name} training completed. Validation accuracy: {val_metrics.get('accuracy', 0):.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
                training_results[model_name] = {'error': str(e)}
        
        return training_results
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray, 
                       training_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all trained models
        
        Args:
            X_test: Test features
            y_test: Test labels
            training_results: Training results
            
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info("Evaluating models...")
        
        evaluation_results = {}
        
        for model_name, results in training_results.items():
            if 'error' in results:
                continue
                
            try:
                model = results['model']
                
                # Make predictions
                test_predictions = model.predict(X_test)
                test_probabilities = model.predict_proba(X_test)
                
                # Calculate metrics
                test_metrics = self.evaluator.compute_metrics(y_test, test_predictions, test_probabilities)
                model.update_metrics(test_metrics, 'test')
                
                # Generate plots
                plot_path = f"outputs/curves/{model_name}_evaluation.png"
                self.evaluator.plot_results(y_test, test_predictions, test_probabilities, plot_path)
                
                # Store results
                evaluation_results[model_name] = {
                    'test_metrics': test_metrics,
                    'predictions': test_predictions,
                    'probabilities': test_probabilities,
                    'feature_importance': results['feature_importance']
                }
                
                self.logger.log_evaluation_results(test_metrics)
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
                evaluation_results[model_name] = {'error': str(e)}
        
        return evaluation_results
    
    def save_results(self, training_results: Dict[str, Any], 
                    evaluation_results: Dict[str, Any]) -> None:
        """
        Save all results
        
        Args:
            training_results: Training results
            evaluation_results: Evaluation results
        """
        self.logger.info("Saving results...")
        
        # Save models
        for model_name, results in training_results.items():
            if 'error' not in results:
                model_path = f"outputs/models/{model_name}_model.pkl"
                results['model'].save_model(model_path)
        
        # Save evaluation results
        results_path = "outputs/reports/evaluation_results.yaml"
        with open(results_path, 'w') as f:
            yaml.dump(evaluation_results, f, default_flow_style=False)
        
        # Generate summary report
        self._generate_summary_report(evaluation_results)
        
        self.logger.info("Results saved successfully")
    
    def _generate_summary_report(self, evaluation_results: Dict[str, Any]) -> None:
        """
        Generate a summary report
        
        Args:
            evaluation_results: Evaluation results
        """
        report_path = "outputs/reports/summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("ML1 Loan Prediction - Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Model performance comparison
            f.write("Model Performance Comparison:\n")
            f.write("-" * 30 + "\n")
            
            model_scores = []
            for model_name, results in evaluation_results.items():
                if 'error' not in results:
                    accuracy = results['test_metrics'].get('accuracy', 0)
                    f1_score = results['test_metrics'].get('f1_score', 0)
                    model_scores.append((model_name, accuracy, f1_score))
                    f.write(f"{model_name}:\n")
                    f.write(f"  Accuracy: {accuracy:.4f}\n")
                    f.write(f"  F1 Score: {f1_score:.4f}\n\n")
            
            # Best model
            if model_scores:
                best_model = max(model_scores, key=lambda x: x[1])
                f.write(f"Best Model: {best_model[0]} (Accuracy: {best_model[1]:.4f})\n")
        
        self.logger.info(f"Summary report saved to {report_path}")
    
    def run_pipeline(self, data_path: str) -> None:
        """
        Run the complete ML pipeline
        
        Args:
            data_path: Path to the data file
        """
        self.logger.info("Starting ML1 Pipeline...")
        
        try:
            # Step 1: Load and validate data
            df = self.load_and_validate_data(data_path)
            
            # Step 2: Preprocess data
            X_train, X_val, y_train, y_val, X_test, y_test = self.preprocess_data(df)
            
            # Step 3: Initialize models
            self.initialize_models()
            
            # Step 4: Train models
            training_results = self.train_models(X_train, y_train, X_val, y_val)
            
            # Step 5: Evaluate models
            evaluation_results = self.evaluate_models(X_test, y_test, training_results)
            
            # Step 6: Save results
            self.save_results(training_results, evaluation_results)
            
            self.logger.info("ML1 Pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
        
        finally:
            # Close logger
            self.logger.close()


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='ML1 Loan Prediction Pipeline')
    parser.add_argument('--data', type=str, default='data/train_u6lujuX_CVtuZ9i.csv',
                       help='Path to training data')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = ML1Pipeline(args.config)
    pipeline.run_pipeline(args.data)


if __name__ == "__main__":
    main()
