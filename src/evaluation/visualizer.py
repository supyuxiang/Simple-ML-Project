"""
Result visualizer for ML1 project
Provides comprehensive visualization capabilities for model evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..core.logger import Logger


class ResultVisualizer:
    """
    Comprehensive result visualizer for ML1 project
    Provides various visualization methods for model evaluation and analysis
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        """
        Initialize the result visualizer
        
        Args:
            config: Visualization configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
        # Visualization settings
        self.plot_style = config.get('plot_style', 'whitegrid')
        self.figure_size = config.get('figure_size', (12, 8))
        self.dpi = config.get('dpi', 300)
        self.color_palette = config.get('color_palette', 'Set2')
        
        # Set plotting style
        plt.style.use('default')
        sns.set_style(self.plot_style)
        sns.set_palette(self.color_palette)
        
        if self.logger:
            self.logger.info("ResultVisualizer initialized")
    
    def plot_model_performance_comparison(self, model_results: Dict[str, Dict[str, Any]],
                                        metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
                                        title: str = "Model Performance Comparison",
                                        save_path: Optional[str] = None) -> None:
        """
        Create comprehensive model performance comparison plots
        
        Args:
            model_results: Dictionary of model results
            metrics: List of metrics to compare
            title: Plot title
            save_path: Path to save the plot
        """
        if not model_results:
            if self.logger:
                self.logger.warning("No model results available for comparison")
            return
        
        # Extract valid results
        valid_results = {name: results for name, results in model_results.items() 
                        if 'error' not in results}
        
        if not valid_results:
            if self.logger:
                self.logger.warning("No valid model results for comparison")
            return
        
        # Create comparison data
        comparison_data = []
        for model_name, results in valid_results.items():
            test_metrics = results.get('test_metrics', {})
            row = {'Model': model_name}
            for metric in metrics:
                row[metric] = test_metrics.get(metric, 0)
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create subplots
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), dpi=self.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            if i < len(axes):
                bars = axes[i].bar(df_comparison['Model'], df_comparison[metric], 
                                 color=plt.cm.Set3(np.linspace(0, 1, len(df_comparison))))
                
                # Add value labels on bars
                for bar, value in zip(bars, df_comparison[metric]):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=9)
                
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_ylabel('Score')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].set_ylim(0, 1.1)
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance_comparison(self, model_results: Dict[str, Dict[str, Any]],
                                         top_k: int = 10,
                                         title: str = "Feature Importance Comparison",
                                         save_path: Optional[str] = None) -> None:
        """
        Compare feature importance across models
        
        Args:
            model_results: Dictionary of model results
            top_k: Number of top features to display
            title: Plot title
            save_path: Path to save the plot
        """
        if not model_results:
            if self.logger:
                self.logger.warning("No model results available for feature importance comparison")
            return
        
        # Extract models with feature importance
        models_with_importance = {}
        for model_name, results in model_results.items():
            if 'error' not in results and 'feature_importance' in results:
                feature_importance = results['feature_importance']
                if feature_importance:
                    models_with_importance[model_name] = feature_importance
        
        if not models_with_importance:
            if self.logger:
                self.logger.warning("No models with feature importance data")
            return
        
        # Create subplots for each model
        n_models = len(models_with_importance)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), dpi=self.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Plot feature importance for each model
        for i, (model_name, feature_importance) in enumerate(models_with_importance.items()):
            if i < len(axes):
                # Get top K features
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
                features, importances = zip(*sorted_features)
                
                bars = axes[i].barh(range(len(features)), importances, 
                                  color=plt.cm.viridis(np.linspace(0, 1, len(features))))
                
                axes[i].set_yticks(range(len(features)))
                axes[i].set_yticklabels(features)
                axes[i].set_xlabel('Importance Score')
                axes[i].set_title(f'{model_name} - Top {top_k} Features')
                axes[i].invert_yaxis()
                
                # Add value labels
                for j, (bar, importance) in enumerate(zip(bars, importances)):
                    axes[i].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                               f'{importance:.3f}', ha='left', va='center', fontsize=8)
        
        # Hide unused subplots
        for i in range(len(models_with_importance), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"Feature importance comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrices(self, model_results: Dict[str, Dict[str, Any]],
                               title: str = "Confusion Matrices Comparison",
                               save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrices for all models
        
        Args:
            model_results: Dictionary of model results
            title: Plot title
            save_path: Path to save the plot
        """
        if not model_results:
            if self.logger:
                self.logger.warning("No model results available for confusion matrices")
            return
        
        # Extract models with predictions
        models_with_predictions = {}
        for model_name, results in model_results.items():
            if 'error' not in results and 'predictions' in results:
                models_with_predictions[model_name] = results['predictions']
        
        if not models_with_predictions:
            if self.logger:
                self.logger.warning("No models with prediction data")
            return
        
        # We need true labels - this should be passed as parameter in real implementation
        # For now, we'll create a placeholder
        if self.logger:
            self.logger.warning("Confusion matrices require true labels - not implemented in this example")
        
        # Create subplots
        n_models = len(models_with_predictions)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), dpi=self.dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Placeholder for confusion matrices
        for i, model_name in enumerate(models_with_predictions.keys()):
            if i < len(axes):
                axes[i].text(0.5, 0.5, f'Confusion Matrix\n{model_name}\n(Requires true labels)', 
                           ha='center', va='center', transform=axes[i].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                axes[i].set_title(f'{model_name}')
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(models_with_predictions), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"Confusion matrices plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, model_results: Dict[str, Dict[str, Any]],
                       y_true: np.ndarray,
                       title: str = "ROC Curves Comparison",
                       save_path: Optional[str] = None) -> None:
        """
        Plot ROC curves for all models
        
        Args:
            model_results: Dictionary of model results
            y_true: True labels
            title: Plot title
            save_path: Path to save the plot
        """
        if not model_results:
            if self.logger:
                self.logger.warning("No model results available for ROC curves")
            return
        
        from sklearn.metrics import roc_curve, roc_auc_score
        
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        
        # Plot ROC curve for each model
        for model_name, results in model_results.items():
            if 'error' not in results and 'probabilities' in results:
                y_proba = results['probabilities']
                
                if y_proba.shape[1] > 1:
                    y_proba_pos = y_proba[:, 1]
                else:
                    y_proba_pos = y_proba.flatten()
                
                fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
                auc_score = roc_auc_score(y_true, y_proba_pos)
                
                plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        # Plot random classifier
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"ROC curves plot saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, model_results: Dict[str, Dict[str, Any]],
                                   y_true: np.ndarray,
                                   title: str = "Precision-Recall Curves Comparison",
                                   save_path: Optional[str] = None) -> None:
        """
        Plot Precision-Recall curves for all models
        
        Args:
            model_results: Dictionary of model results
            y_true: True labels
            title: Plot title
            save_path: Path to save the plot
        """
        if not model_results:
            if self.logger:
                self.logger.warning("No model results available for PR curves")
            return
        
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        
        # Plot PR curve for each model
        for model_name, results in model_results.items():
            if 'error' not in results and 'probabilities' in results:
                y_proba = results['probabilities']
                
                if y_proba.shape[1] > 1:
                    y_proba_pos = y_proba[:, 1]
                else:
                    y_proba_pos = y_proba.flatten()
                
                precision, recall, _ = precision_recall_curve(y_true, y_proba_pos)
                ap_score = average_precision_score(y_true, y_proba_pos)
                
                plt.plot(recall, precision, lw=2, label=f'{model_name} (AP = {ap_score:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"Precision-Recall curves plot saved to {save_path}")
        
        plt.show()
    
    def create_comprehensive_dashboard(self, model_results: Dict[str, Dict[str, Any]],
                                     data_info: Dict[str, Any],
                                     y_true: Optional[np.ndarray] = None,
                                     title: str = "ML1 Comprehensive Dashboard",
                                     save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive dashboard with all visualizations
        
        Args:
            model_results: Dictionary of model results
            data_info: Data information dictionary
            y_true: True labels (optional)
            title: Dashboard title
            save_path: Path to save the dashboard
        """
        if not model_results:
            if self.logger:
                self.logger.warning("No model results available for dashboard")
            return
        
        # Create large figure for dashboard
        fig = plt.figure(figsize=(20, 16), dpi=self.dpi)
        fig.suptitle(title, fontsize=20, fontweight='bold')
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        # 1. Model Performance Comparison (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_performance_bars(model_results, ax1)
        
        # 2. Data Overview (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_data_overview(data_info, ax2)
        
        # 3. Feature Importance (second row, left)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_best_model_feature_importance(model_results, ax3)
        
        # 4. Metrics Heatmap (second row, right)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_metrics_heatmap(model_results, ax4)
        
        # 5. ROC Curves (third row, left)
        ax5 = fig.add_subplot(gs[2, :2])
        if y_true is not None:
            self._plot_roc_curves_subplot(model_results, y_true, ax5)
        else:
            ax5.text(0.5, 0.5, 'ROC Curves\n(Requires true labels)', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('ROC Curves')
            ax5.axis('off')
        
        # 6. Precision-Recall Curves (third row, right)
        ax6 = fig.add_subplot(gs[2, 2:])
        if y_true is not None:
            self._plot_pr_curves_subplot(model_results, y_true, ax6)
        else:
            ax6.text(0.5, 0.5, 'PR Curves\n(Requires true labels)', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Precision-Recall Curves')
            ax6.axis('off')
        
        # 7. Training Summary (bottom row)
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_training_summary(model_results, ax7)
        
        # Save dashboard if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"Comprehensive dashboard saved to {save_path}")
        
        plt.show()
    
    def _plot_performance_bars(self, model_results: Dict[str, Dict[str, Any]], ax) -> None:
        """Helper method to plot performance bars"""
        valid_results = {name: results for name, results in model_results.items() 
                        if 'error' not in results}
        
        if not valid_results:
            ax.text(0.5, 0.5, 'No valid results', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Performance')
            return
        
        models = list(valid_results.keys())
        accuracies = [results.get('test_metrics', {}).get('accuracy', 0) for results in valid_results.values()]
        f1_scores = [results.get('test_metrics', {}).get('f1_score', 0) for results in valid_results.values()]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_data_overview(self, data_info: Dict[str, Any], ax) -> None:
        """Helper method to plot data overview"""
        if not data_info:
            ax.text(0.5, 0.5, 'No data info', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Data Overview')
            return
        
        info_text = f"""
        Dataset Shape: {data_info.get('shape', 'N/A')}
        Features: {data_info.get('n_features', 'N/A')}
        Samples: {data_info.get('n_samples', 'N/A')}
        Missing Values: {data_info.get('missing_count', 'N/A')}
        Target Classes: {data_info.get('n_classes', 'N/A')}
        """
        ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.set_title('Dataset Overview')
        ax.axis('off')
    
    def _plot_best_model_feature_importance(self, model_results: Dict[str, Dict[str, Any]], ax) -> None:
        """Helper method to plot best model feature importance"""
        # Find best model
        best_model = None
        best_score = 0
        for model_name, results in model_results.items():
            if 'error' not in results:
                test_metrics = results.get('test_metrics', {})
                accuracy = test_metrics.get('accuracy', 0)
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = results
        
        if best_model and 'feature_importance' in best_model:
            feature_importance = best_model['feature_importance']
            if feature_importance:
                # Get top 10 features
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                features, importances = zip(*sorted_features)
                
                bars = ax.barh(range(len(features)), importances, color='skyblue', alpha=0.7)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features)
                ax.set_xlabel('Importance')
                ax.set_title('Top 10 Feature Importance (Best Model)')
                ax.invert_yaxis()
                
                # Add value labels
                for bar, importance in zip(bars, importances):
                    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                           f'{importance:.3f}', ha='left', va='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No feature importance data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance')
            ax.axis('off')
    
    def _plot_metrics_heatmap(self, model_results: Dict[str, Dict[str, Any]], ax) -> None:
        """Helper method to plot metrics heatmap"""
        valid_results = {name: results for name, results in model_results.items() 
                        if 'error' not in results}
        
        if not valid_results:
            ax.text(0.5, 0.5, 'No valid results', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Metrics Heatmap')
            return
        
        # Create metrics matrix
        metrics_data = []
        for model_name, results in valid_results.items():
            test_metrics = results.get('test_metrics', {})
            row = {'Model': model_name}
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                row[metric] = test_metrics.get(metric, 0)
            metrics_data.append(row)
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            metrics_matrix = df_metrics.set_index('Model')[['accuracy', 'precision', 'recall', 'f1_score']]
            sns.heatmap(metrics_matrix, annot=True, cmap='YlOrRd', ax=ax, fmt='.3f')
            ax.set_title('Model Metrics Heatmap')
        else:
            ax.text(0.5, 0.5, 'No metrics data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Metrics Heatmap')
            ax.axis('off')
    
    def _plot_roc_curves_subplot(self, model_results: Dict[str, Dict[str, Any]], y_true: np.ndarray, ax) -> None:
        """Helper method to plot ROC curves in subplot"""
        from sklearn.metrics import roc_curve, roc_auc_score
        
        for model_name, results in model_results.items():
            if 'error' not in results and 'probabilities' in results:
                y_proba = results['probabilities']
                
                if y_proba.shape[1] > 1:
                    y_proba_pos = y_proba[:, 1]
                else:
                    y_proba_pos = y_proba.flatten()
                
                fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
                auc_score = roc_auc_score(y_true, y_proba_pos)
                
                ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_pr_curves_subplot(self, model_results: Dict[str, Dict[str, Any]], y_true: np.ndarray, ax) -> None:
        """Helper method to plot PR curves in subplot"""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        for model_name, results in model_results.items():
            if 'error' not in results and 'probabilities' in results:
                y_proba = results['probabilities']
                
                if y_proba.shape[1] > 1:
                    y_proba_pos = y_proba[:, 1]
                else:
                    y_proba_pos = y_proba.flatten()
                
                precision, recall, _ = precision_recall_curve(y_true, y_proba_pos)
                ap_score = average_precision_score(y_true, y_proba_pos)
                
                ax.plot(recall, precision, lw=2, label=f'{model_name} (AP = {ap_score:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_training_summary(self, model_results: Dict[str, Dict[str, Any]], ax) -> None:
        """Helper method to plot training summary"""
        valid_results = {name: results for name, results in model_results.items() 
                        if 'error' not in results}
        
        if not valid_results:
            summary_text = "No valid model results available"
        else:
            best_model_name = max(valid_results.items(), 
                                key=lambda x: x[1].get('test_metrics', {}).get('accuracy', 0))[0]
            best_accuracy = max([r.get('test_metrics', {}).get('accuracy', 0) 
                               for r in valid_results.values()])
            
            summary_text = f"""
            ML1 Loan Prediction System - Training Summary
            
            • Total Models Trained: {len(valid_results)}
            • Best Model: {best_model_name}
            • Best Accuracy: {best_accuracy:.4f}
            • Training Completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            • System Status: Ready for Production
            """
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
        ax.set_title('Training Summary')
        ax.axis('off')
