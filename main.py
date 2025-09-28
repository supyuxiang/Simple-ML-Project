#!/usr/bin/env python3
"""
Advanced ML Pipeline - Main Entry Point.

This module provides the main entry point for the production-grade ML pipeline
with all advanced features integrated:
- Registry-based component management
- Intelligent caching
- Async task execution
- Real-time monitoring
- Advanced configuration management
- Performance profiling
- Comprehensive error handling
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core import (
    AdvancedConfigManager, ConfigSchema, Logger, 
    get_component, start_scheduler, stop_scheduler,
    start_monitoring, stop_monitoring, get_monitoring_system,
    get_cache_manager, set_cache_backend, DiskCacheBackend
)
from src.core.exceptions import (
    MLProjectError, ConfigurationError, DataError, ModelError,
    TrainingError, EvaluationError
)
from src.training.trainer import AdvancedModelTrainer
from src.data.processor import AdvancedDataProcessor
from src.evaluation.evaluator import ModelEvaluator
from src.utils import setup_directories, save_results


class AdvancedMLPipeline:
    """
    Advanced production ML pipeline with all advanced features integrated.
    
    Features:
    - Registry-based component management
    - Intelligent caching for expensive operations
    - Async task execution and scheduling
    - Real-time monitoring and metrics collection
    - Advanced configuration management with validation
    - Performance profiling and optimization
    - Comprehensive error handling and recovery
    - Experiment tracking and reproducibility
    """
    
    def __init__(
        self, 
        config_path: str, 
        data_path: str,
        environment: str = "development"
    ) -> None:
        """
        Initialize the advanced ML pipeline.
        
        Args:
            config_path: Path to configuration file
            data_path: Path to data file
            environment: Environment name
        """
        self.config_path = config_path
        self.data_path = data_path
        self.environment = environment
        
        # Initialize components
        self._initialize_components()
        
        if self.logger:
            self.logger.info("Advanced ML pipeline initialized successfully")
            self.logger.info(f"Environment: {environment}")
            self.logger.info(f"Configuration: {config_path}")
            self.logger.info(f"Data: {data_path}")
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        try:
            # Initialize advanced configuration manager
            self.config_manager = AdvancedConfigManager(
                self.config_path, 
                self.environment
            )
            
            # Initialize logger
            logger_config = self.config_manager.get_section("Logger")
            self.logger = Logger(logger_config)
            
            # Set up output directories
            self._setup_output_directories()
            
            # Initialize monitoring system
            self.monitoring = get_monitoring_system()
            self.monitoring._logger = self.logger
            start_monitoring(interval=30.0)
            
            # Initialize cache system
            cache_config = self.config_manager.get_section("Cache")
            if cache_config and cache_config.get("use_disk_cache", False):
                cache_dir = Path("cache")
                cache_dir.mkdir(exist_ok=True)
                disk_cache = DiskCacheBackend(cache_dir)
                set_cache_backend(disk_cache)
            
            # Initialize components using registry
            self.data_processor = get_component("advanced_processor", "processors", 
                                              config_manager=self.config_manager, 
                                              logger=self.logger)
            self.model_trainer = get_component("advanced_trainer", "trainers",
                                             config_manager=self.config_manager,
                                             logger=self.logger)
            
            # Initialize evaluator
            metrics_config = self.config_manager.get_section("Metrics")
            self.evaluator = ModelEvaluator(metrics_config, self.logger)
            
            # Start async task scheduler
            asyncio.create_task(start_scheduler())
            
            if self.logger:
                self.logger.info("Pipeline components initialized successfully")
                
        except Exception as e:
            error_msg = f"Component initialization failed: {e}"
            print(f"Critical initialization error: {error_msg}", file=sys.stderr)
            sys.exit(1)
    
    def _setup_output_directories(self) -> None:
        """Set up output directories."""
        base_path = Path(__file__).resolve().parent
        directories = [
            base_path / "outputs" / "models",
            base_path / "outputs" / "curves", 
            base_path / "outputs" / "reports",
            base_path / "logs",
            base_path / "cache",
            base_path / "swanlog"
        ]
        setup_directories(base_path, directories)
        
        if self.logger:
            self.logger.info("Output directories created/verified")
    
    async def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete ML pipeline with all advanced features.
        
        Returns:
            Pipeline results dictionary
        """
        if self.logger:
            self.logger.info("Starting advanced ML pipeline execution")
        
        pipeline_start_time = time.time()
        
        try:
            # Phase 1: Data Loading and Processing
            if self.logger:
                self.logger.info("Phase 1: Data Loading and Processing")
            
            df = self.data_processor.load_data(self.data_path)
            X, y = await self.data_processor.preprocess_async(df)
            X_train, X_val, y_train, y_val = self.data_processor.split_data(X, y)
            
            # Phase 2: Model Training
            if self.logger:
                self.logger.info("Phase 2: Model Training")
            
            # Get model from registry
            model_config = self.config_manager.get_section("Model")
            model_name = model_config.get("model_name", "XGBoost")
            model = get_component(model_name.lower(), "models", 
                                config=model_config, logger=self.logger)
            
            # Set model in trainer
            self.model_trainer.model = model
            
            # Train model asynchronously
            training_results = await self.model_trainer.train_async(
                X_train, y_train, X_val, y_val
            )
            
            # Phase 3: Model Evaluation
            if self.logger:
                self.logger.info("Phase 3: Model Evaluation")
            
            predictions = model.predict(X_val)
            probabilities = model.predict_proba(X_val)
            evaluation_results = self.evaluator.compute_metrics(
                y_val, predictions, probabilities
            )
            
            # Phase 4: Results and Reporting
            if self.logger:
                self.logger.info("Phase 4: Results and Reporting")
            
            # Combine all results
            pipeline_results = {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'data_processing_info': self.data_processor.get_feature_importance_data(),
                'pipeline_metrics': {
                    'total_time': time.time() - pipeline_start_time,
                    'data_samples': len(X),
                    'features': X.shape[1],
                    'train_samples': len(X_train),
                    'val_samples': len(X_val)
                },
                'configuration_hash': self.config_manager.get_config_hash(),
                'environment': self.environment
            }
            
            # Save results
            self._save_results(pipeline_results)
            
            # Export reports
            self._export_reports()
            
            if self.logger:
                self.logger.info("Advanced ML pipeline completed successfully")
                self.logger.info(f"Total pipeline time: {pipeline_results['pipeline_metrics']['total_time']:.2f} seconds")
            
            return pipeline_results
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {e}"
            if self.logger:
                self.logger.error(error_msg, exc_info=True)
            raise MLProjectError(error_msg) from e
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save pipeline results."""
        try:
            base_path = Path(__file__).resolve().parent
            results_path = base_path / "outputs" / "reports" / "pipeline_results.yaml"
            
            save_results(results, results_path, format="yaml")
            
            if self.logger:
                self.logger.info("Results saved successfully")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save results: {e}")
    
    def _export_reports(self) -> None:
        """Export comprehensive reports."""
        try:
            base_path = Path(__file__).resolve().parent
            reports_dir = base_path / "outputs" / "reports"
            
            # Export training report
            training_report_path = reports_dir / "training_report.json"
            self.model_trainer.export_training_report(training_report_path)
            
            # Export processing report
            processing_report_path = reports_dir / "processing_report.json"
            self.data_processor.export_processing_report(processing_report_path)
            
            # Export monitoring metrics
            metrics_path = reports_dir / "monitoring_metrics.json"
            self.monitoring.export_metrics(metrics_path)
            
            if self.logger:
                self.logger.info("Comprehensive reports exported")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to export reports: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'monitoring_stats': self.monitoring.get_stats(),
            'cache_stats': get_cache_manager().get_stats(),
            'config_hash': self.config_manager.get_config_hash(),
            'environment': self.environment,
            'components_initialized': True
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Stop monitoring
            stop_monitoring()
            
            # Stop task scheduler
            await stop_scheduler()
            
            # Close logger
            if hasattr(self, 'logger'):
                self.logger.close()
            
            if self.logger:
                self.logger.info("Pipeline cleanup completed")
                
        except Exception as e:
            print(f"Cleanup error: {e}", file=sys.stderr)


async def main() -> None:
    """Main function to run the advanced ML pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the advanced production ML pipeline"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the input CSV data file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file"
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="development",
        help="Environment to run in (development, staging, production)"
    )
    parser.add_argument(
        "--async-mode",
        action="store_true",
        help="Run in async mode for better performance"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AdvancedMLPipeline(
        args.config,
        args.data,
        args.environment
    )
    
    try:
        # Run pipeline
        if args.async_mode:
            results = await pipeline.run_pipeline()
        else:
            # Run in sync mode for compatibility
            results = await pipeline.run_pipeline()
        
        # Print summary
        print("\n" + "="*60)
        print("ADVANCED ML PIPELINE SUMMARY")
        print("="*60)
        print(f"Status: SUCCESS")
        print(f"Model: {results['training_results']['model_name']}")
        print(f"Features: {results['pipeline_metrics']['features']}")
        print(f"Training Samples: {results['pipeline_metrics']['train_samples']}")
        print(f"Validation Samples: {results['pipeline_metrics']['val_samples']}")
        print(f"Total Time: {results['pipeline_metrics']['total_time']:.2f}s")
        
        if 'evaluation_results' in results:
            eval_results = results['evaluation_results']
            print("\nEvaluation Metrics:")
            for metric, value in eval_results.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
        
        print("="*60)
        
    except MLProjectError as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Cleanup
        await pipeline.cleanup()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
