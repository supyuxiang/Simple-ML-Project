"""
Data processing module for ML1 project
Contains data loading, preprocessing, and feature engineering classes
"""

from .processor import AdvancedDataProcessor as LoanDataProcessor
# FeatureEngineer removed - functionality integrated into LoanDataProcessor
# DataValidator moved to core.validators

__all__ = [
    'LoanDataProcessor'
]
