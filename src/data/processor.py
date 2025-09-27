"""
Data processor for loan prediction dataset
Handles loading, cleaning, and preprocessing of loan data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

from ..core.base import BaseDataProcessor
from ..core.logger import Logger


class LoanDataProcessor(BaseDataProcessor):
    """
    Data processor for loan prediction dataset
    Handles comprehensive data preprocessing pipeline
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        """
        Initialize the data processor
        
        Args:
            config: Data processing configuration
            logger: Logger instance
        """
        super().__init__(config)
        self.logger = logger
        self.label_encoders = {}
        self.scalers = {}
        self.imputers = {}
        self.feature_names = None
        self.target_name = 'Loan_Status'
        
        # Data processing parameters
        self.test_size = config.get('test_size', 0.2)
        self.random_state = config.get('random_seed', 42)
        self.validation_size = config.get('validation_size', 0.2)
        
        # Feature engineering parameters
        self.create_features = config.get('create_features', True)
        self.scale_features = config.get('scale_features', True)
        self.scaling_method = config.get('scaling_method', 'standard')  # 'standard', 'minmax', 'none'
        
        # Missing value handling
        self.missing_strategy = config.get('missing_strategy', 'smart')  # 'smart', 'drop', 'fill'
        self.categorical_imputer = config.get('categorical_imputer', 'most_frequent')
        self.numerical_imputer = config.get('numerical_imputer', 'median')
        
        if self.logger:
            self.logger.info("LoanDataProcessor initialized")
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load loan data from CSV file
        
        Args:
            data_path: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_csv(data_path)
            
            if self.logger:
                self.logger.info(f"Data loaded successfully from {data_path}")
                self.logger.info(f"Data shape: {df.shape}")
                self.logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load data: {e}")
            raise
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data analysis
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing data analysis results
        """
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        analysis['categorical_columns'] = categorical_cols
        
        # Numerical columns analysis
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        analysis['numerical_columns'] = numerical_cols
        
        # Target variable analysis
        if self.target_name in df.columns:
            target_analysis = {
                'value_counts': df[self.target_name].value_counts().to_dict(),
                'value_counts_pct': df[self.target_name].value_counts(normalize=True).to_dict(),
                'is_balanced': len(df[self.target_name].unique()) > 1
            }
            analysis['target_analysis'] = target_analysis
        
        # Statistical summary
        if numerical_cols:
            analysis['numerical_summary'] = df[numerical_cols].describe().to_dict()
        
        if self.logger:
            self.logger.info("Data analysis completed")
            self.logger.info(f"Missing values: {analysis['missing_values']}")
            self.logger.info(f"Categorical columns: {categorical_cols}")
            self.logger.info(f"Numerical columns: {numerical_cols}")
        
        return analysis
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df_processed = df.copy()
        
        # Identify column types
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.missing_strategy == 'drop':
            # Drop rows with any missing values
            initial_shape = df_processed.shape
            df_processed = df_processed.dropna()
            if self.logger:
                self.logger.info(f"Dropped {initial_shape[0] - df_processed.shape[0]} rows with missing values")
        
        elif self.missing_strategy == 'fill':
            # Fill missing values
            # Categorical columns
            for col in categorical_cols:
                if df_processed[col].isnull().sum() > 0:
                    if self.categorical_imputer == 'most_frequent':
                        mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
                        df_processed[col] = df_processed[col].fillna(mode_value)
                    else:
                        df_processed[col] = df_processed[col].fillna('Unknown')
            
            # Numerical columns
            for col in numerical_cols:
                if df_processed[col].isnull().sum() > 0:
                    if self.numerical_imputer == 'median':
                        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                    elif self.numerical_imputer == 'mean':
                        df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                    elif self.numerical_imputer == 'knn':
                        if col not in self.imputers:
                            self.imputers[col] = KNNImputer(n_neighbors=5)
                        df_processed[col] = self.imputers[col].fit_transform(df_processed[[col]]).flatten()
        
        elif self.missing_strategy == 'smart':
            # Smart handling based on missing percentage
            for col in df_processed.columns:
                missing_pct = df_processed[col].isnull().sum() / len(df_processed) * 100
                
                if missing_pct > 50:
                    # Drop column if more than 50% missing
                    df_processed = df_processed.drop(columns=[col])
                    if self.logger:
                        self.logger.warning(f"Dropped column {col} due to {missing_pct:.1f}% missing values")
                
                elif missing_pct > 0:
                    # Fill missing values
                    if col in categorical_cols:
                        mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
                        df_processed[col] = df_processed[col].fillna(mode_value)
                    else:
                        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        if self.logger:
            remaining_missing = df_processed.isnull().sum().sum()
            self.logger.info(f"Missing values after processing: {remaining_missing}")
        
        return df_processed
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable from encoding if it exists
        if self.target_name in categorical_cols:
            categorical_cols.remove(self.target_name)
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                # Handle unseen categories
                unique_values = set(df_encoded[col].astype(str).unique())
                known_values = set(self.label_encoders[col].classes_)
                unseen_values = unique_values - known_values
                
                if unseen_values:
                    if self.logger:
                        self.logger.warning(f"Unseen categories in {col}: {unseen_values}")
                    # Replace unseen categories with most frequent category
                    most_frequent = self.label_encoders[col].classes_[0]
                    df_encoded[col] = df_encoded[col].astype(str).replace(list(unseen_values), most_frequent)
                
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        if self.logger:
            self.logger.info(f"Encoded {len(categorical_cols)} categorical features")
        
        return df_encoded
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features for better model performance
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        if not self.create_features:
            return df
        
        df_features = df.copy()
        
        # Income-related features
        if 'ApplicantIncome' in df_features.columns and 'CoapplicantIncome' in df_features.columns:
            # Total income
            df_features['TotalIncome'] = df_features['ApplicantIncome'] + df_features['CoapplicantIncome']
            
            # Income ratio
            df_features['IncomeRatio'] = df_features['ApplicantIncome'] / (df_features['CoapplicantIncome'] + 1)
            
            # Income categories
            df_features['IncomeCategory'] = pd.cut(
                df_features['TotalIncome'], 
                bins=[0, 3000, 6000, 10000, float('inf')], 
                labels=['Low', 'Medium', 'High', 'VeryHigh']
            )
            df_features['IncomeCategory'] = df_features['IncomeCategory'].astype(str)
        
        # Loan-related features
        if 'LoanAmount' in df_features.columns and 'TotalIncome' in df_features.columns:
            # Loan to income ratio
            df_features['LoanToIncomeRatio'] = df_features['LoanAmount'] / (df_features['TotalIncome'] + 1)
            
            # EMI (assuming 8% annual interest rate)
            if 'Loan_Amount_Term' in df_features.columns:
                monthly_rate = 0.08 / 12
                df_features['EMI'] = df_features['LoanAmount'] * monthly_rate * (1 + monthly_rate)**df_features['Loan_Amount_Term'] / ((1 + monthly_rate)**df_features['Loan_Amount_Term'] - 1)
                df_features['EMI'] = df_features['EMI'].fillna(0)
        
        # Credit history features
        if 'Credit_History' in df_features.columns:
            # Credit history categories
            df_features['CreditCategory'] = df_features['Credit_History'].apply(
                lambda x: 'Good' if x == 1 else 'Bad' if x == 0 else 'Unknown'
            )
        
        # Property area features
        if 'Property_Area' in df_features.columns:
            # Property area categories
            df_features['PropertyCategory'] = df_features['Property_Area'].apply(
                lambda x: 'Urban' if x == 'Urban' else 'Rural' if x == 'Rural' else 'Semiurban'
            )
        
        if self.logger:
            new_features = [col for col in df_features.columns if col not in df.columns]
            self.logger.info(f"Created {len(new_features)} new features: {new_features}")
        
        return df_features
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with scaled features
        """
        if not self.scale_features or self.scaling_method == 'none':
            return df
        
        df_scaled = df.copy()
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target variable from scaling
        if self.target_name in numerical_cols:
            numerical_cols.remove(self.target_name)
        
        for col in numerical_cols:
            if col not in self.scalers:
                if self.scaling_method == 'standard':
                    self.scalers[col] = StandardScaler()
                elif self.scaling_method == 'minmax':
                    self.scalers[col] = MinMaxScaler()
                
                df_scaled[col] = self.scalers[col].fit_transform(df_scaled[[col]]).flatten()
            else:
                df_scaled[col] = self.scalers[col].transform(df_scaled[[col]]).flatten()
        
        if self.logger:
            self.logger.info(f"Scaled {len(numerical_cols)} numerical features using {self.scaling_method} scaling")
        
        return df_scaled
    
    def preprocess(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline
        
        Args:
            data: Raw data
            
        Returns:
            Tuple of (features, labels)
        """
        if self.logger:
            self.logger.info("Starting data preprocessing pipeline")
        
        # Step 1: Handle missing values
        df_processed = self.handle_missing_values(data)
        
        # Step 2: Create new features
        df_processed = self.create_features(df_processed)
        
        # Step 3: Encode categorical features
        df_processed = self.encode_categorical_features(df_processed)
        
        # Step 4: Scale features
        df_processed = self.scale_features(df_processed)
        
        # Step 5: Separate features and target
        if self.target_name in df_processed.columns:
            # Encode target variable
            if self.target_name not in self.label_encoders:
                self.label_encoders[self.target_name] = LabelEncoder()
                y = self.label_encoders[self.target_name].fit_transform(df_processed[self.target_name])
            else:
                y = self.label_encoders[self.target_name].transform(df_processed[self.target_name])
            
            # Remove target and ID columns from features
            feature_cols = [col for col in df_processed.columns 
                          if col not in [self.target_name, 'Loan_ID']]
            X = df_processed[feature_cols].values
            
            # Store feature names
            self.feature_names = feature_cols
            
        else:
            # No target variable (test data)
            feature_cols = [col for col in df_processed.columns if col != 'Loan_ID']
            X = df_processed[feature_cols].values
            y = None
            self.feature_names = feature_cols
        
        if self.logger:
            self.logger.info(f"Preprocessing completed. Features shape: {X.shape}")
            if y is not None:
                self.logger.info(f"Target shape: {y.shape}")
                self.logger.info(f"Target distribution: {np.bincount(y)}")
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and validation sets
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.validation_size,
            random_state=self.random_state,
            stratify=y
        )
        
        if self.logger:
            self.logger.info(f"Data split completed:")
            self.logger.info(f"  Train set: {X_train.shape[0]} samples")
            self.logger.info(f"  Validation set: {X_val.shape[0]} samples")
        
        return X_train, X_val, y_train, y_val
    
    def get_feature_importance_data(self) -> Dict[str, Any]:
        """
        Get feature importance information
        
        Returns:
            Dictionary containing feature information
        """
        return {
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names) if self.feature_names else 0,
            'label_encoders': list(self.label_encoders.keys()),
            'scalers': list(self.scalers.keys()),
            'imputers': list(self.imputers.keys())
        }
    
    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Inverse transform encoded target variable
        
        Args:
            y_encoded: Encoded target variable
            
        Returns:
            Original target variable
        """
        if self.target_name in self.label_encoders:
            return self.label_encoders[self.target_name].inverse_transform(y_encoded)
        else:
            return y_encoded
