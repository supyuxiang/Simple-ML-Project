"""
Advanced data processor with production-grade features.

This module provides a comprehensive data processing system with:
- Registry-based component management
- Intelligent caching for expensive operations
- Async processing capabilities
- Real-time monitoring and metrics
- Advanced configuration management
- Performance profiling and optimization
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from ..core.interfaces import BaseDataProcessor
from ..core.logger import Logger
from ..core.registry import register_component, get_component
from ..core.cache import cached, get_cache_manager
from ..core.async_tasks import submit_function, TaskPriority
from ..core.monitoring import profile_function, add_metric, get_monitoring_system
from ..core.advanced_config import AdvancedConfigManager
from ..core.validators import DataValidator
from ..core.constants import (
    FEATURE_ENGINEERING, TARGET_COLUMN, ID_COLUMN
)
from ..core.exceptions import DataError, PreprocessingError


@register_component("advanced_processor", "processors")
class AdvancedDataProcessor(BaseDataProcessor):
    """
    Production-grade data processor with advanced features.
    
    Features:
    - Component registry integration
    - Intelligent caching for expensive operations
    - Async processing capabilities
    - Real-time monitoring and metrics
    - Performance profiling
    - Advanced configuration management
    - Data quality assessment
    """
    
    def __init__(
        self, 
        config_manager: AdvancedConfigManager,
        logger: Optional[Logger] = None
    ):
        """
        Initialize the advanced data processor.
        
        Args:
            config_manager: Advanced configuration manager
            logger: Logger instance
        """
        super().__init__(config_manager.get_section("Data"))
        self.config_manager = config_manager
        self.logger = logger
        self.monitoring = get_monitoring_system()
        self.cache_manager = get_cache_manager()
        self.validator = DataValidator()
        
        # Configuration
        self.data_config = config_manager.get_section("Data")
        self.validation_config = config_manager.get_section("Validation")
        
        # Processing parameters
        self.test_size = self.data_config.get('test_size', 0.2)
        self.random_seed = self.data_config.get('random_seed', 42)
        self.missing_strategy = self.data_config.get('missing_strategy', 'smart')
        self.scale_features = self.data_config.get('scale_features', True)
        self.scaling_method = self.data_config.get('scaling_method', 'standard')
        self.create_features = self.data_config.get('create_features', True)
        self.feature_selection = self.data_config.get('feature_selection', True)
        self.n_features_select = self.data_config.get('n_features_select', 10)
        
        # Processing state
        self.processing_history: List[Dict[str, Any]] = []
        self.feature_names: List[str] = []
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.imputers: Dict[str, Any] = {}
        
        if self.logger:
            self.logger.info("AdvancedDataProcessor initialized with production features")
    
    @profile_function
    @cached(ttl=7200)  # Cache for 2 hours
    def load_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data with caching and monitoring.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        data_path = Path(data_path)
        
        if self.logger:
            self.logger.info(f"Loading data from {data_path}")
        
        start_time = time.time()
        
        try:
            # Load data
            if data_path.suffix.lower() == '.csv':
                df = pd.read_csv(data_path)
            elif data_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(data_path)
            elif data_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(data_path)
            else:
                raise DataError(f"Unsupported file format: {data_path.suffix}")
            
            # Validate data
            validation_results = self.validator.validate_dataframe(df)
            
            if not validation_results['is_valid']:
                if self.logger:
                    self.logger.warning(f"Data validation issues: {validation_results['errors']}")
            
            # Add metrics
            loading_time = time.time() - start_time
            add_metric("data.loading_time", loading_time)
            add_metric("data.samples", len(df))
            add_metric("data.features", len(df.columns))
            add_metric("data.quality_score", validation_results['quality_score'])
            add_metric("data.missing_percentage", df.isnull().sum().sum() / df.size * 100)
            
            if self.logger:
                self.logger.info(f"Data loaded successfully: {len(df)} samples, {len(df.columns)} features")
                self.logger.info(f"Data quality score: {validation_results['quality_score']:.3f}")
            
            return df
            
        except Exception as e:
            add_metric("data.loading_errors", 1)
            error_msg = f"Data loading failed: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise DataError(error_msg) from e
    
    @profile_function
    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Comprehensive data preprocessing with monitoring.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features, targets)
        """
        if self.logger:
            self.logger.info("Starting comprehensive data preprocessing pipeline")
        
        start_time = time.time()
        
        try:
            # Step 1: Handle missing values
            if self.logger:
                self.logger.info("Step 1/6: Handling missing values")
            
            df_processed = self._handle_missing_values(df)
            add_metric("preprocessing.missing_values_handled", df.isnull().sum().sum() - df_processed.isnull().sum().sum())
            
            # Step 2: Feature engineering
            if self.logger:
                self.logger.info("Step 2/6: Feature engineering")
            
            df_features = self._create_features(df_processed)
            add_metric("preprocessing.features_created", len(df_features.columns) - len(df_processed.columns))
            
            # Step 3: Encode categorical features
            if self.logger:
                self.logger.info("Step 3/6: Encoding categorical features")
            
            df_encoded = self._encode_categorical_features(df_features)
            add_metric("preprocessing.categorical_features_encoded", len(self.encoders))
            
            # Step 4: Scale features
            if self.logger:
                self.logger.info("Step 4/6: Scaling features")
            
            df_scaled = self._scale_features(df_encoded)
            add_metric("preprocessing.features_scaled", len(self.scalers))
            
            # Step 5: Separate features and target
            if self.logger:
                self.logger.info("Step 5/6: Separating features and target")
            
            X, y = self._separate_features_target(df_scaled)
            
            # Step 6: Final validation
            if self.logger:
                self.logger.info("Step 6/6: Final validation")
            
            self._validate_processed_data(X, y)
            
            # Store processing metadata
            processing_time = time.time() - start_time
            processing_metadata = {
                'timestamp': time.time(),
                'samples': len(X),
                'features': X.shape[1],
                'processing_time': processing_time,
                'feature_names': self.feature_names.copy(),
                'scalers_used': list(self.scalers.keys()),
                'encoders_used': list(self.encoders.keys())
            }
            self.processing_history.append(processing_metadata)
            
            # Add final metrics
            add_metric("preprocessing.total_time", processing_time)
            add_metric("preprocessing.final_features", X.shape[1])
            add_metric("preprocessing.final_samples", len(X))
            
            if self.logger:
                self.logger.info("Data preprocessing pipeline completed successfully")
                self.logger.info(f"Final features shape: {X.shape}")
                self.logger.info(f"Target shape: {y.shape}")
                self.logger.info(f"Target distribution: {np.bincount(y)}")
            
            return X, y
            
        except Exception as e:
            add_metric("preprocessing.errors", 1)
            error_msg = f"Data preprocessing failed: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise PreprocessingError(error_msg) from e
    
    @cached(ttl=3600)  # Cache for 1 hour
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with intelligent strategies."""
        df_processed = df.copy()
        
        # Get missing value statistics
        missing_stats = df_processed.isnull().sum()
        missing_percentage = (missing_stats / len(df_processed)) * 100
        
        if self.logger:
            self.logger.info(f"Missing values after processing: {df_processed.isnull().sum().sum()}")
        
        # Apply missing value strategy
        if self.missing_strategy == 'smart':
            # Smart imputation based on data type and missing percentage
            for column in df_processed.columns:
                if df_processed[column].isnull().any():
                    if df_processed[column].dtype in ['object', 'category']:
                        # Categorical: use most frequent
                        df_processed[column].fillna(df_processed[column].mode()[0], inplace=True)
                    else:
                        # Numerical: use median for high missing percentage, mean for low
                        if missing_percentage[column] > 20:
                            df_processed[column].fillna(df_processed[column].median(), inplace=True)
                        else:
                            df_processed[column].fillna(df_processed[column].mean(), inplace=True)
        
        elif self.missing_strategy == 'knn':
            # KNN imputation for numerical columns
            numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                knn_imputer = KNNImputer(n_neighbors=5)
                df_processed[numerical_cols] = knn_imputer.fit_transform(df_processed[numerical_cols])
        
        return df_processed
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features with monitoring."""
        if not self.create_features:
            if self.logger:
                self.logger.debug("Feature engineering disabled, skipping feature creation")
            return df
        
        if self.logger:
            self.logger.info("Starting feature engineering")
        
        df_features = df.copy()
        original_columns = set(df.columns)
        
        try:
            # Get feature engineering parameters from constants
            income_bins = FEATURE_ENGINEERING['income_bins']
            income_labels = FEATURE_ENGINEERING['income_labels']
            interest_rate = FEATURE_ENGINEERING['interest_rate']
            
            # Income-related features
            if 'ApplicantIncome' in df_features.columns and 'CoapplicantIncome' in df_features.columns:
                # Total income
                df_features['TotalIncome'] = (
                    df_features['ApplicantIncome'] + df_features['CoapplicantIncome']
                )
                
                # Income ratio
                df_features['IncomeRatio'] = (
                    df_features['ApplicantIncome'] / (df_features['CoapplicantIncome'] + 1)
                )
                
                # Income categories
                df_features['IncomeCategory'] = pd.cut(
                    df_features['TotalIncome'], 
                    bins=income_bins,
                    labels=income_labels,
                    include_lowest=True
                ).astype(str)
                
                if self.logger:
                    self.logger.debug("Created income-related features")
            
            # Loan-related features
            if 'LoanAmount' in df_features.columns:
                if 'TotalIncome' in df_features.columns:
                    # Loan to income ratio
                    df_features['LoanToIncomeRatio'] = (
                        df_features['LoanAmount'] / (df_features['TotalIncome'] + 1)
                    )
                
                # EMI calculation
                if 'Loan_Amount_Term' in df_features.columns:
                    monthly_rate = interest_rate / 12
                    df_features['EMI'] = (
                        df_features['LoanAmount'] * monthly_rate *
                        (1 + monthly_rate) ** df_features['Loan_Amount_Term'] /
                        ((1 + monthly_rate) ** df_features['Loan_Amount_Term'] - 1)
                    )
                    df_features['EMI'] = df_features['EMI'].fillna(0)
                    
                    if self.logger:
                        self.logger.debug("Created loan-related features")
            
            # Credit history features
            if 'Credit_History' in df_features.columns:
                df_features['CreditCategory'] = df_features['Credit_History'].apply(
                    lambda x: 'Good' if x == 1 else 'Bad' if x == 0 else 'Unknown'
                )
                
                if self.logger:
                    self.logger.debug("Created credit history features")
            
            # Property area features
            if 'Property_Area' in df_features.columns:
                df_features['PropertyCategory'] = df_features['Property_Area'].apply(
                    lambda x: 'Urban' if x == 'Urban' else 'Rural' if x == 'Rural' else 'Semiurban'
                )
                
                if self.logger:
                    self.logger.debug("Created property area features")
                
            # Log created features
            new_features = [col for col in df_features.columns if col not in original_columns]
            if new_features and self.logger:
                self.logger.info(f"Created {len(new_features)} new features: {new_features}")
            
            return df_features
            
        except Exception as e:
            error_msg = f"Feature engineering failed: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise PreprocessingError(error_msg)
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features with monitoring."""
        df_encoded = df.copy()
        
        # Identify categorical columns
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col not in [TARGET_COLUMN, ID_COLUMN]]
        
        for col in categorical_cols:
            if df_encoded[col].dtype == 'object':
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = encoder
        
        if self.logger:
            self.logger.info(f"Encoded {len(categorical_cols)} categorical features")
        
        return df_encoded
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features with monitoring."""
        if not self.scale_features:
            return df
        
        df_scaled = df.copy()
        
        # Identify numerical columns to scale
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in [TARGET_COLUMN, ID_COLUMN]]
        
        if self.scaling_method == 'standard':
            scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            return df_scaled
        
        # Scale features
        df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
        self.scalers[self.scaling_method] = scaler
        
        if self.logger:
            self.logger.info(f"Scaled {len(numerical_cols)} numerical features using {self.scaling_method} scaling")
        
        return df_scaled
    
    def _separate_features_target(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Separate features and target with validation."""
        # Remove ID column if present
        feature_cols = [col for col in df.columns if col not in [TARGET_COLUMN, ID_COLUMN]]
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df[TARGET_COLUMN].values
        
        # Encode target if it's categorical
        if df[TARGET_COLUMN].dtype == 'object':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
            self.encoders[TARGET_COLUMN] = target_encoder
        
        return X, y
    
    def _validate_processed_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate processed data."""
        # Check for NaN or infinite values
        if np.isnan(X).any():
            raise PreprocessingError("Features contain NaN values")
        if np.isinf(X).any():
            raise PreprocessingError("Features contain infinite values")
        if np.isnan(y).any():
            raise PreprocessingError("Target contains NaN values")
        if np.isinf(y).any():
            raise PreprocessingError("Target contains infinite values")
        
        # Check data shapes
        if X.shape[0] != y.shape[0]:
            raise PreprocessingError("Feature and target sample counts don't match")
        
        if self.logger:
            self.logger.info(f"Data validation passed. Final features shape: {X.shape}, Target shape: {y.shape}")
    
    @profile_function
    def split_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data with monitoring.
        
        Args:
            X: Features
            y: Targets
            test_size: Test set size
            random_state: Random state
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        test_size = test_size or self.test_size
        random_state = random_state or self.random_seed
        
        if self.logger:
            self.logger.info("Data split completed:")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        # Add metrics
        add_metric("data.train_samples", len(X_train))
        add_metric("data.val_samples", len(X_val))
        add_metric("data.train_features", X_train.shape[1])
        add_metric("data.val_features", X_val.shape[1])
        
        if self.logger:
            self.logger.info(f"  Train set: {len(X_train)} samples")
            self.logger.info(f"  Validation set: {len(X_val)} samples")
        
        return X_train, X_val, y_train, y_val
    
    async def preprocess_async(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data asynchronously.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (features, targets)
        """
        if self.logger:
            self.logger.info("Starting async data preprocessing...")
        
        # Submit preprocessing as async task
        task_id = submit_function(
            self.preprocess,
            args=(df,),
            priority=TaskPriority.HIGH
        )
        
        # Wait for completion
        from ..core.async_tasks import get_scheduler
        scheduler = get_scheduler()
        result = await scheduler.wait_for_task(task_id)
        
        if result.status.value == "completed":
            return result.result
        else:
            raise PreprocessingError(f"Async preprocessing failed: {result.error}")
    
    def get_processing_history(self) -> List[Dict[str, Any]]:
        """Get processing history."""
        return self.processing_history.copy()
    
    def get_feature_importance_data(self) -> Dict[str, Any]:
        """Get feature importance data."""
        return {
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'scalers_used': list(self.scalers.keys()),
            'encoders_used': list(self.encoders.keys())
        }
    
    def export_processing_report(self, filepath: Union[str, Path]) -> None:
        """Export comprehensive processing report."""
        import json
        
        report = {
            'processing_history': self.processing_history,
            'feature_importance_data': self.get_feature_importance_data(),
            'configuration': {
                'data_config': self.data_config,
                'validation_config': self.validation_config,
                'config_hash': self.config_manager.get_config_hash()
            },
            'monitoring_metrics': {
                'loading_time': self.monitoring.get_metric_summary("data.loading_time"),
                'preprocessing_time': self.monitoring.get_metric_summary("preprocessing.total_time"),
                'quality_score': self.monitoring.get_metric_summary("data.quality_score")
            },
            'performance_profiles': self.monitoring.get_performance_profiles()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        if self.logger:
            self.logger.info(f"Processing report exported to: {filepath}")
