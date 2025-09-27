"""
Data validation module for loan prediction
Contains data quality checks and validation functions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

from ..core.logger import Logger


class DataValidator:
    """
    Data validation and quality checks for loan prediction dataset
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        """
        Initialize the data validator
        
        Args:
            config: Validation configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.validation_results = {}
        
        # Validation parameters
        self.missing_threshold = config.get('missing_threshold', 0.5)
        self.outlier_method = config.get('outlier_method', 'iqr')  # 'iqr', 'zscore', 'isolation'
        self.outlier_threshold = config.get('outlier_threshold', 3.0)
        self.correlation_threshold = config.get('correlation_threshold', 0.95)
        self.variance_threshold = config.get('variance_threshold', 0.0)
        
        if self.logger:
            self.logger.info("DataValidator initialized")
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality validation
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing validation results
        """
        if self.logger:
            self.logger.info("Starting data quality validation")
        
        validation_results = {
            'basic_info': self._get_basic_info(df),
            'missing_values': self._check_missing_values(df),
            'duplicates': self._check_duplicates(df),
            'data_types': self._check_data_types(df),
            'outliers': self._check_outliers(df),
            'correlations': self._check_correlations(df),
            'variance': self._check_variance(df),
            'target_distribution': self._check_target_distribution(df),
            'data_consistency': self._check_data_consistency(df)
        }
        
        # Overall quality score
        validation_results['quality_score'] = self._calculate_quality_score(validation_results)
        
        self.validation_results = validation_results
        
        if self.logger:
            self.logger.info(f"Data quality validation completed. Quality score: {validation_results['quality_score']:.2f}")
        
        return validation_results
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get basic information about the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing basic information
        """
        return {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict()
        }
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for missing values
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing missing value analysis
        """
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        high_missing_cols = missing_percentages[missing_percentages > self.missing_threshold * 100].index.tolist()
        
        return {
            'missing_counts': missing_counts.to_dict(),
            'missing_percentages': missing_percentages.to_dict(),
            'total_missing': missing_counts.sum(),
            'high_missing_columns': high_missing_cols,
            'has_high_missing': len(high_missing_cols) > 0
        }
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for duplicate rows
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing duplicate analysis
        """
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        return {
            'duplicate_count': duplicate_count,
            'duplicate_percentage': duplicate_percentage,
            'has_duplicates': duplicate_count > 0
        }
    
    def _check_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data types consistency
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing data type analysis
        """
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Check for mixed types in categorical columns
        mixed_type_cols = []
        for col in categorical_cols:
            if df[col].apply(lambda x: isinstance(x, (int, float)) and pd.isna(x) == False).any():
                mixed_type_cols.append(col)
        
        return {
            'categorical_columns': categorical_cols,
            'numerical_columns': numerical_cols,
            'mixed_type_columns': mixed_type_cols,
            'has_mixed_types': len(mixed_type_cols) > 0
        }
    
    def _check_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for outliers in numerical columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing outlier analysis
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_info = {}
        
        for col in numerical_cols:
            if self.outlier_method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            
            elif self.outlier_method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > self.outlier_threshold][col]
            
            else:
                outliers = pd.Series(dtype=float)
            
            outlier_info[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(df)) * 100,
                'outlier_values': outliers.tolist() if len(outliers) > 0 else []
            }
        
        total_outliers = sum(info['outlier_count'] for info in outlier_info.values())
        
        return {
            'outlier_info': outlier_info,
            'total_outliers': total_outliers,
            'has_outliers': total_outliers > 0
        }
    
    def _check_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for high correlations between features
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing correlation analysis
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            return {'high_correlations': [], 'has_high_correlations': False}
        
        corr_matrix = df[numerical_cols].corr()
        
        # Find high correlations (excluding diagonal)
        high_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > self.correlation_threshold:
                    high_correlations.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_correlations,
            'has_high_correlations': len(high_correlations) > 0
        }
    
    def _check_variance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for low variance features
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing variance analysis
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        low_variance_cols = []
        
        for col in numerical_cols:
            variance = df[col].var()
            if variance < self.variance_threshold:
                low_variance_cols.append({
                    'column': col,
                    'variance': variance
                })
        
        return {
            'low_variance_columns': low_variance_cols,
            'has_low_variance': len(low_variance_cols) > 0
        }
    
    def _check_target_distribution(self, df: pd.DataFrame, target_col: str = 'Loan_Status') -> Dict[str, Any]:
        """
        Check target variable distribution
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Dictionary containing target distribution analysis
        """
        if target_col not in df.columns:
            return {'error': f'Target column {target_col} not found'}
        
        target_counts = df[target_col].value_counts()
        target_percentages = df[target_col].value_counts(normalize=True) * 100
        
        # Check for class imbalance
        max_percentage = target_percentages.max()
        is_imbalanced = max_percentage > 80  # Consider imbalanced if one class > 80%
        
        return {
            'value_counts': target_counts.to_dict(),
            'percentages': target_percentages.to_dict(),
            'is_imbalanced': is_imbalanced,
            'imbalance_ratio': target_percentages.max() / target_percentages.min()
        }
    
    def _check_data_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data consistency and logical constraints
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing consistency analysis
        """
        consistency_issues = []
        
        # Check for negative values in positive-only columns
        positive_only_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        for col in positive_only_cols:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    consistency_issues.append(f"Negative values in {col}: {negative_count}")
        
        # Check for unrealistic values
        if 'ApplicantIncome' in df.columns:
            unrealistic_income = (df['ApplicantIncome'] > 100000).sum()
            if unrealistic_income > 0:
                consistency_issues.append(f"Unrealistic income values: {unrealistic_income}")
        
        if 'LoanAmount' in df.columns:
            unrealistic_loan = (df['LoanAmount'] > 1000000).sum()
            if unrealistic_loan > 0:
                consistency_issues.append(f"Unrealistic loan amounts: {unrealistic_loan}")
        
        # Check for logical inconsistencies
        if 'Credit_History' in df.columns and 'Loan_Status' in df.columns:
            # Check if people with bad credit history got loans
            bad_credit_approved = ((df['Credit_History'] == 0) & (df['Loan_Status'] == 'Y')).sum()
            if bad_credit_approved > 0:
                consistency_issues.append(f"Bad credit but approved loans: {bad_credit_approved}")
        
        return {
            'consistency_issues': consistency_issues,
            'has_issues': len(consistency_issues) > 0,
            'issue_count': len(consistency_issues)
        }
    
    def _calculate_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """
        Calculate overall data quality score
        
        Args:
            validation_results: Validation results dictionary
            
        Returns:
            Quality score between 0 and 1
        """
        score = 1.0
        
        # Deduct points for various issues
        if validation_results['missing_values']['has_high_missing']:
            score -= 0.2
        
        if validation_results['duplicates']['has_duplicates']:
            score -= 0.1
        
        if validation_results['data_types']['has_mixed_types']:
            score -= 0.1
        
        if validation_results['outliers']['has_outliers']:
            score -= 0.1
        
        if validation_results['correlations']['has_high_correlations']:
            score -= 0.1
        
        if validation_results['variance']['has_low_variance']:
            score -= 0.1
        
        if validation_results['data_consistency']['has_issues']:
            score -= 0.2
        
        return max(0.0, score)
    
    def generate_validation_report(self, validation_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a comprehensive validation report
        
        Args:
            validation_results: Validation results (optional)
            
        Returns:
            Formatted validation report
        """
        if validation_results is None:
            validation_results = self.validation_results
        
        if not validation_results:
            return "No validation results available"
        
        report = []
        report.append("=" * 60)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 60)
        
        # Basic information
        basic_info = validation_results.get('basic_info', {})
        report.append(f"\nDataset Shape: {basic_info.get('shape', 'N/A')}")
        report.append(f"Memory Usage: {basic_info.get('memory_usage', 0):.2f} MB")
        report.append(f"Quality Score: {validation_results.get('quality_score', 0):.2f}")
        
        # Missing values
        missing_info = validation_results.get('missing_values', {})
        report.append(f"\nMissing Values:")
        report.append(f"  Total missing: {missing_info.get('total_missing', 0)}")
        if missing_info.get('has_high_missing', False):
            report.append(f"  High missing columns: {missing_info.get('high_missing_columns', [])}")
        
        # Duplicates
        duplicate_info = validation_results.get('duplicates', {})
        report.append(f"\nDuplicates:")
        report.append(f"  Duplicate rows: {duplicate_info.get('duplicate_count', 0)}")
        report.append(f"  Duplicate percentage: {duplicate_info.get('duplicate_percentage', 0):.2f}%")
        
        # Outliers
        outlier_info = validation_results.get('outliers', {})
        report.append(f"\nOutliers:")
        report.append(f"  Total outliers: {outlier_info.get('total_outliers', 0)}")
        
        # Data consistency
        consistency_info = validation_results.get('data_consistency', {})
        report.append(f"\nData Consistency:")
        report.append(f"  Issues found: {consistency_info.get('issue_count', 0)}")
        for issue in consistency_info.get('consistency_issues', []):
            report.append(f"    - {issue}")
        
        # Target distribution
        target_info = validation_results.get('target_distribution', {})
        if 'value_counts' in target_info:
            report.append(f"\nTarget Distribution:")
            for class_name, count in target_info['value_counts'].items():
                percentage = target_info['percentages'].get(class_name, 0)
                report.append(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def get_recommendations(self, validation_results: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Get data quality improvement recommendations
        
        Args:
            validation_results: Validation results (optional)
            
        Returns:
            List of recommendations
        """
        if validation_results is None:
            validation_results = self.validation_results
        
        recommendations = []
        
        # Missing values recommendations
        missing_info = validation_results.get('missing_values', {})
        if missing_info.get('has_high_missing', False):
            recommendations.append("Consider dropping columns with high missing values or use advanced imputation methods")
        
        # Duplicate recommendations
        duplicate_info = validation_results.get('duplicates', {})
        if duplicate_info.get('has_duplicates', False):
            recommendations.append("Remove duplicate rows to avoid data leakage")
        
        # Outlier recommendations
        outlier_info = validation_results.get('outliers', {})
        if outlier_info.get('has_outliers', False):
            recommendations.append("Investigate and handle outliers appropriately")
        
        # Correlation recommendations
        correlation_info = validation_results.get('correlations', {})
        if correlation_info.get('has_high_correlations', False):
            recommendations.append("Consider removing highly correlated features to reduce multicollinearity")
        
        # Target distribution recommendations
        target_info = validation_results.get('target_distribution', {})
        if target_info.get('is_imbalanced', False):
            recommendations.append("Consider using techniques to handle class imbalance (SMOTE, class weights, etc.)")
        
        # Data consistency recommendations
        consistency_info = validation_results.get('data_consistency', {})
        if consistency_info.get('has_issues', False):
            recommendations.append("Review and fix data consistency issues")
        
        return recommendations
