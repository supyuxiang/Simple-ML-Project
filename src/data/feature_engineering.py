"""
Feature engineering module for loan prediction
Contains advanced feature engineering techniques
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from ..core.logger import Logger


class FeatureEngineer:
    """
    Advanced feature engineering for loan prediction
    Implements various feature engineering techniques
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        """
        Initialize the feature engineer
        
        Args:
            config: Feature engineering configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.transformers = {}
        self.selectors = {}
        self.feature_names = None
        
        # Feature engineering parameters
        self.polynomial_degree = config.get('polynomial_degree', 2)
        self.power_transform = config.get('power_transform', False)
        self.quantile_transform = config.get('quantile_transform', False)
        self.feature_selection = config.get('feature_selection', True)
        self.n_features_select = config.get('n_features_select', 10)
        self.pca_components = config.get('pca_components', None)
        self.clustering_features = config.get('clustering_features', False)
        self.n_clusters = config.get('n_clusters', 3)
        
        if self.logger:
            self.logger.info("FeatureEngineer initialized")
    
    def create_polynomial_features(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Create polynomial features
        
        Args:
            X: Input features
            feature_names: Original feature names
            
        Returns:
            Tuple of (polynomial features, new feature names)
        """
        if self.polynomial_degree <= 1:
            return X, feature_names
        
        poly = PolynomialFeatures(
            degree=self.polynomial_degree,
            include_bias=False,
            interaction_only=False
        )
        
        X_poly = poly.fit_transform(X)
        
        # Generate feature names
        poly_feature_names = poly.get_feature_names_out(feature_names)
        
        if self.logger:
            self.logger.info(f"Created polynomial features: {X.shape[1]} -> {X_poly.shape[1]}")
        
        return X_poly, poly_feature_names.tolist()
    
    def apply_power_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply power transformation to make data more Gaussian-like
        
        Args:
            X: Input features
            
        Returns:
            Power transformed features
        """
        if not self.power_transform:
            return X
        
        if 'power_transformer' not in self.transformers:
            self.transformers['power_transformer'] = PowerTransformer(method='yeo-johnson')
            X_transformed = self.transformers['power_transformer'].fit_transform(X)
        else:
            X_transformed = self.transformers['power_transformer'].transform(X)
        
        if self.logger:
            self.logger.info("Applied power transformation")
        
        return X_transformed
    
    def apply_quantile_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply quantile transformation to make data uniformly distributed
        
        Args:
            X: Input features
            
        Returns:
            Quantile transformed features
        """
        if not self.quantile_transform:
            return X
        
        if 'quantile_transformer' not in self.transformers:
            self.transformers['quantile_transformer'] = QuantileTransformer(
                output_distribution='uniform',
                random_state=42
            )
            X_transformed = self.transformers['quantile_transformer'].fit_transform(X)
        else:
            X_transformed = self.transformers['quantile_transformer'].transform(X)
        
        if self.logger:
            self.logger.info("Applied quantile transformation")
        
        return X_transformed
    
    def create_clustering_features(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Create clustering-based features
        
        Args:
            X: Input features
            feature_names: Original feature names
            
        Returns:
            Tuple of (features with clustering, new feature names)
        """
        if not self.clustering_features:
            return X, feature_names
        
        # Create clustering features
        clustering_features = []
        clustering_names = []
        
        # K-means clustering
        if 'kmeans' not in self.transformers:
            self.transformers['kmeans'] = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
            cluster_labels = self.transformers['kmeans'].fit_predict(X)
        else:
            cluster_labels = self.transformers['kmeans'].predict(X)
        
        # Add cluster labels as features
        clustering_features.append(cluster_labels.reshape(-1, 1))
        clustering_names.append('cluster_label')
        
        # Add distance to cluster centers
        if hasattr(self.transformers['kmeans'], 'cluster_centers_'):
            distances = self.transformers['kmeans'].transform(X)
            for i in range(self.n_clusters):
                clustering_features.append(distances[:, i].reshape(-1, 1))
                clustering_names.append(f'distance_to_cluster_{i}')
        
        # Combine with original features
        if clustering_features:
            X_clustered = np.hstack([X] + clustering_features)
            new_feature_names = feature_names + clustering_names
        else:
            X_clustered = X
            new_feature_names = feature_names
        
        if self.logger:
            self.logger.info(f"Created clustering features: {X.shape[1]} -> {X_clustered.shape[1]}")
        
        return X_clustered, new_feature_names
    
    def select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Select most important features
        
        Args:
            X: Input features
            y: Target variable
            feature_names: Feature names
            
        Returns:
            Tuple of (selected features, selected feature names)
        """
        if not self.feature_selection:
            return X, feature_names
        
        n_features = min(self.n_features_select, X.shape[1])
        
        # Use mutual information for feature selection
        if 'feature_selector' not in self.selectors:
            self.selectors['feature_selector'] = SelectKBest(
                score_func=mutual_info_classif,
                k=n_features
            )
            X_selected = self.selectors['feature_selector'].fit_transform(X, y)
            selected_indices = self.selectors['feature_selector'].get_support(indices=True)
        else:
            X_selected = self.selectors['feature_selector'].transform(X)
            selected_indices = self.selectors['feature_selector'].get_support(indices=True)
        
        selected_feature_names = [feature_names[i] for i in selected_indices]
        
        if self.logger:
            self.logger.info(f"Feature selection: {X.shape[1]} -> {X_selected.shape[1]} features")
            self.logger.info(f"Selected features: {selected_feature_names}")
        
        return X_selected, selected_feature_names
    
    def apply_pca(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Apply Principal Component Analysis
        
        Args:
            X: Input features
            feature_names: Feature names
            
        Returns:
            Tuple of (PCA features, PCA feature names)
        """
        if self.pca_components is None:
            return X, feature_names
        
        n_components = min(self.pca_components, X.shape[1])
        
        if 'pca' not in self.transformers:
            self.transformers['pca'] = PCA(
                n_components=n_components,
                random_state=42
            )
            X_pca = self.transformers['pca'].fit_transform(X)
        else:
            X_pca = self.transformers['pca'].transform(X)
        
        # Generate PCA feature names
        pca_feature_names = [f'PC_{i+1}' for i in range(n_components)]
        
        if self.logger:
            explained_variance = self.transformers['pca'].explained_variance_ratio_
            self.logger.info(f"PCA applied: {X.shape[1]} -> {X_pca.shape[1]} components")
            self.logger.info(f"Explained variance ratio: {explained_variance[:5]}")  # Top 5 components
        
        return X_pca, pca_feature_names
    
    def create_interaction_features(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Create interaction features between important features
        
        Args:
            X: Input features
            feature_names: Feature names
            
        Returns:
            Tuple of (features with interactions, new feature names)
        """
        # Select top features for interactions (to avoid explosion of features)
        n_top_features = min(5, X.shape[1])
        top_indices = np.argsort(np.var(X, axis=0))[-n_top_features:]
        
        interaction_features = []
        interaction_names = []
        
        # Create pairwise interactions
        for i in range(len(top_indices)):
            for j in range(i+1, len(top_indices)):
                idx1, idx2 = top_indices[i], top_indices[j]
                interaction = X[:, idx1] * X[:, idx2]
                interaction_features.append(interaction.reshape(-1, 1))
                interaction_names.append(f"{feature_names[idx1]}_x_{feature_names[idx2]}")
        
        if interaction_features:
            X_interactions = np.hstack([X] + interaction_features)
            new_feature_names = feature_names + interaction_names
        else:
            X_interactions = X
            new_feature_names = feature_names
        
        if self.logger:
            self.logger.info(f"Created interaction features: {X.shape[1]} -> {X_interactions.shape[1]}")
        
        return X_interactions, new_feature_names
    
    def engineer_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Complete feature engineering pipeline
        
        Args:
            X: Input features
            y: Target variable
            feature_names: Feature names
            
        Returns:
            Tuple of (engineered features, new feature names)
        """
        if self.logger:
            self.logger.info("Starting feature engineering pipeline")
        
        current_X = X.copy()
        current_names = feature_names.copy()
        
        # Step 1: Power transformation
        current_X = self.apply_power_transform(current_X)
        
        # Step 2: Quantile transformation
        current_X = self.apply_quantile_transform(current_X)
        
        # Step 3: Clustering features
        current_X, current_names = self.create_clustering_features(current_X, current_names)
        
        # Step 4: Interaction features
        current_X, current_names = self.create_interaction_features(current_X, current_names)
        
        # Step 5: Polynomial features
        current_X, current_names = self.create_polynomial_features(current_X, current_names)
        
        # Step 6: Feature selection
        current_X, current_names = self.select_features(current_X, y, current_names)
        
        # Step 7: PCA (optional)
        current_X, current_names = self.apply_pca(current_X, current_names)
        
        self.feature_names = current_names
        
        if self.logger:
            self.logger.info(f"Feature engineering completed: {X.shape[1]} -> {current_X.shape[1]} features")
        
        return current_X, current_names
    
    def get_feature_importance_scores(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Args:
            X: Input features
            y: Target variable
            feature_names: Feature names
            
        Returns:
            Dictionary of feature importance scores
        """
        # Use Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importance_scores = dict(zip(feature_names, rf.feature_importances_))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
        
        if self.logger:
            self.logger.info("Top 10 most important features:")
            for i, (feature, score) in enumerate(list(sorted_importance.items())[:10]):
                self.logger.info(f"  {i+1}. {feature}: {score:.4f}")
        
        return sorted_importance
    
    def get_transformation_info(self) -> Dict[str, Any]:
        """
        Get information about applied transformations
        
        Returns:
            Dictionary containing transformation information
        """
        info = {
            'transformers': list(self.transformers.keys()),
            'selectors': list(self.selectors.keys()),
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names) if self.feature_names else 0
        }
        
        # Add specific transformation details
        if 'pca' in self.transformers:
            pca = self.transformers['pca']
            info['pca_explained_variance_ratio'] = pca.explained_variance_ratio_.tolist()
            info['pca_n_components'] = pca.n_components_
        
        if 'feature_selector' in self.selectors:
            selector = self.selectors['feature_selector']
            info['feature_selector_scores'] = selector.scores_.tolist()
            info['feature_selector_k'] = selector.k
        
        return info
