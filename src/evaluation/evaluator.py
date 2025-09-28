"""
Model evaluator for ML1 project
Handles comprehensive model evaluation and metrics computation
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..core.interfaces import BaseEvaluator
from ..core.logger import Logger


class ModelEvaluator(BaseEvaluator):
    """
    Comprehensive model evaluator with advanced metrics and visualizations
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        """
        Initialize the evaluator
        
        Args:
            config: Evaluation configuration
            logger: Logger instance
        """
        super().__init__(config)
        self.logger = logger
        
        # Metrics to compute
        self.metrics_list = config.get('metrics_list', [
            'accuracy', 'precision', 'recall', 'f1_score', 
            'roc_auc', 'confusion_matrix', 'classification_report'
        ])
        
        # Visualization settings
        self.plot_style = config.get('plot_style', 'whitegrid')
        self.figure_size = config.get('figure_size', (12, 8))
        self.dpi = config.get('dpi', 300)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_style(self.plot_style)
        
        if self.logger:
            self.logger.info("ModelEvaluator initialized")
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        if 'accuracy' in self.metrics_list:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        if 'precision' in self.metrics_list:
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        
        if 'recall' in self.metrics_list:
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        
        if 'f1_score' in self.metrics_list:
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # ROC AUC (requires probabilities)
        if 'roc_auc' in self.metrics_list and y_proba is not None:
            try:
                if y_proba.shape[1] > 1:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics['roc_auc'] = 0.0
        
        # Average Precision (requires probabilities)
        if 'average_precision' in self.metrics_list and y_proba is not None:
            try:
                if y_proba.shape[1] > 1:
                    metrics['average_precision'] = average_precision_score(y_true, y_proba[:, 1])
                else:
                    metrics['average_precision'] = average_precision_score(y_true, y_proba)
            except ValueError:
                metrics['average_precision'] = 0.0
        
        # Per-class metrics
        if 'precision_per_class' in self.metrics_list:
            precision_per_class = precision_score(y_true, y_pred, average=None)
            metrics['precision_per_class'] = precision_per_class.tolist()
        
        if 'recall_per_class' in self.metrics_list:
            recall_per_class = recall_score(y_true, y_pred, average=None)
            metrics['recall_per_class'] = recall_per_class.tolist()
        
        if 'f1_per_class' in self.metrics_list:
            f1_per_class = f1_score(y_true, y_pred, average=None)
            metrics['f1_per_class'] = f1_per_class.tolist()
        
        if self.logger:
            self.logger.info(f"Computed {len(metrics)} metrics")
        
        return metrics
    
    def plot_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    y_proba: Optional[np.ndarray] = None, save_path: Optional[str] = None) -> None:
        """
        Create comprehensive evaluation plots
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            save_path: Path to save the plot (optional)
        """
        if self.logger:
            self.logger.info("Creating evaluation plots...")
        
        # Create figure with subplots
        n_plots = 2
        if y_proba is not None:
            n_plots += 2  # Add ROC and PR curves
        
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(y_true, y_pred, axes[0, 0])
        
        # 2. Classification Report
        self._plot_classification_report(y_true, y_pred, axes[0, 1])
        
        # 3. ROC Curve (if probabilities available)
        if y_proba is not None:
            self._plot_roc_curve(y_true, y_proba, axes[1, 0])
            
            # 4. Precision-Recall Curve
            self._plot_precision_recall_curve(y_true, y_proba, axes[1, 1])
        else:
            # Hide unused subplots
            axes[1, 0].set_visible(False)
            axes[1, 1].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"Evaluation plots saved to {save_path}")
        
        plt.show()
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, ax) -> None:
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    def _plot_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, ax) -> None:
        """
        Plot classification report as heatmap
        """
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Extract metrics for heatmap
        metrics_df = pd.DataFrame(report).iloc[:-1, :-1]  # Exclude support and accuracy
        
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='Blues', ax=ax)
        ax.set_title('Classification Report')
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, ax) -> None:
        """
        Plot ROC curve
        """
        if y_proba.shape[1] > 1:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba.flatten()
        
        fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
        auc_score = roc_auc_score(y_true, y_proba_pos)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
    
    def _plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray, ax) -> None:
        """
        Plot Precision-Recall curve
        """
        if y_proba.shape[1] > 1:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba.flatten()
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba_pos)
        ap_score = average_precision_score(y_true, y_proba_pos)
        
        ax.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {ap_score:.2f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]], 
                      save_path: Optional[str] = None) -> None:
        """
        Compare multiple models
        
        Args:
            model_results: Dictionary of model results
            save_path: Path to save the comparison plot
        """
        if self.logger:
            self.logger.info("Creating model comparison plots...")
        
        # Extract metrics for comparison
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        comparison_data = []
        for model_name, results in model_results.items():
            if 'error' not in results:
                model_metrics = results.get('test_metrics', {})
                row = {'Model': model_name}
                for metric in metrics_to_compare:
                    row[metric] = model_metrics.get(metric, 0)
                comparison_data.append(row)
        
        if not comparison_data:
            self.logger.warning("No valid model results for comparison")
            return
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.dpi)
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Plot each metric
        for i, metric in enumerate(metrics_to_compare):
            row = i // 3
            col = i % 3
            
            if row < 2 and col < 3:
                ax = axes[row, col]
                sns.barplot(data=df_comparison, x='Model', y=metric, ax=ax)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplot
        if len(metrics_to_compare) < 6:
            axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"Model comparison plots saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_proba: Optional[np.ndarray] = None,
                                 model_name: str = "Model") -> str:
        """
        Generate a comprehensive evaluation report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            model_name: Name of the model
            
        Returns:
            Formatted evaluation report
        """
        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred, y_proba)
        
        # Generate report
        report = []
        report.append("=" * 60)
        report.append(f"EVALUATION REPORT - {model_name.upper()}")
        report.append("=" * 60)
        
        # Basic metrics
        report.append(f"\nBasic Metrics:")
        report.append(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        report.append(f"  Precision: {metrics.get('precision', 0):.4f}")
        report.append(f"  Recall:    {metrics.get('recall', 0):.4f}")
        report.append(f"  F1 Score:  {metrics.get('f1_score', 0):.4f}")
        
        # Advanced metrics
        if 'roc_auc' in metrics:
            report.append(f"  ROC AUC:   {metrics.get('roc_auc', 0):.4f}")
        
        if 'average_precision' in metrics:
            report.append(f"  Avg Precision: {metrics.get('average_precision', 0):.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        report.append(f"\nConfusion Matrix:")
        report.append(f"  [[{cm[0,0]:3d}, {cm[0,1]:3d}]")
        report.append(f"   [{cm[1,0]:3d}, {cm[1,1]:3d}]]")
        
        # Classification report
        report.append(f"\nDetailed Classification Report:")
        report.append(classification_report(y_true, y_pred))
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of available metrics
        
        Returns:
            Dictionary containing metrics information
        """
        return {
            'available_metrics': self.metrics_list,
            'plot_style': self.plot_style,
            'figure_size': self.figure_size,
            'dpi': self.dpi
        }
