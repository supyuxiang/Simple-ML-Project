"""
Visualization utilities for ML1 project
Contains plotting functions for data analysis and model evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set default style
plt.style.use('default')
sns.set_style('whitegrid')


def plot_feature_importance(feature_importance: Dict[str, float], 
                          top_k: int = 15,
                          title: str = "Feature Importance",
                          figsize: Tuple[int, int] = (10, 8),
                          save_path: Optional[str] = None,
                          logger: Optional[Any] = None) -> None:
    """
    Plot feature importance
    
    Args:
        feature_importance: Dictionary of feature importance scores
        top_k: Number of top features to display
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
    """
    if not feature_importance:
        if logger:
            logger.warning("No feature importance data available")
        return
    
    # Get top K features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    if not sorted_features:
        if logger:
            logger.warning("No features to plot")
        return
    
    features, importances = zip(*sorted_features)
    
    # Create plot
    plt.figure(figsize=figsize)
    bars = plt.barh(range(len(features)), importances, color='skyblue', alpha=0.7)
    
    # Customize plot
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.gca().invert_yaxis()  # Highest importance at top
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, importances)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def plot_training_history(training_history: Dict[str, Any],
                         metrics: List[str] = ['loss', 'accuracy'],
                         title: str = "Training History",
                         figsize: Tuple[int, int] = (12, 6),
                         save_path: Optional[str] = None,
                         logger: Optional[Any] = None) -> None:
    """
    Plot training history
    
    Args:
        training_history: Dictionary containing training history
        metrics: List of metrics to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
    """
    if not training_history:
        if logger:
            logger.warning("No training history data available")
        return
    
    # Create subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric in training_history:
            epochs = range(1, len(training_history[metric]) + 1)
            axes[i].plot(epochs, training_history[metric], 'b-', label=f'Training {metric}')
            
            # Add validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in training_history:
                axes[i].plot(epochs, training_history[val_metric], 'r-', label=f'Validation {metric}')
            
            axes[i].set_title(f'{metric.title()} Over Time')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.title())
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, f'No {metric} data available', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{metric.title()} - No Data')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_data_distribution(df: pd.DataFrame,
                          target_column: str = 'Loan_Status',
                          figsize: Tuple[int, int] = (15, 10),
                          save_path: Optional[str] = None) -> None:
    """
    Plot data distribution and statistics
    
    Args:
        df: DataFrame to analyze
        target_column: Target column name
        figsize: Figure size
        save_path: Path to save the plot
    """
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Target distribution
    if target_column in df.columns:
        target_counts = df[target_column].value_counts()
        axes[0, 0].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title(f'{target_column} Distribution')
    
    # 2. Missing values heatmap
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if len(missing_data) > 0:
        axes[0, 1].bar(range(len(missing_data)), missing_data.values)
        axes[0, 1].set_xticks(range(len(missing_data)))
        axes[0, 1].set_xticklabels(missing_data.index, rotation=45)
        axes[0, 1].set_title('Missing Values by Column')
        axes[0, 1].set_ylabel('Count')
    else:
        axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
        axes[0, 1].set_title('Missing Values - None')
    
    # 3. Numerical columns distribution
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        # Plot first numerical column
        col = numerical_cols[0]
        axes[0, 2].hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue')
        axes[0, 2].set_title(f'{col} Distribution')
        axes[0, 2].set_xlabel(col)
        axes[0, 2].set_ylabel('Frequency')
    else:
        axes[0, 2].text(0.5, 0.5, 'No Numerical Columns', ha='center', va='center')
        axes[0, 2].set_title('Numerical Distribution - None')
    
    # 4. Categorical columns distribution
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        # Plot first categorical column
        col = categorical_cols[0]
        value_counts = df[col].value_counts()
        axes[1, 0].bar(range(len(value_counts)), value_counts.values)
        axes[1, 0].set_xticks(range(len(value_counts)))
        axes[1, 0].set_xticklabels(value_counts.index, rotation=45)
        axes[1, 0].set_title(f'{col} Distribution')
        axes[1, 0].set_ylabel('Count')
    else:
        axes[1, 0].text(0.5, 0.5, 'No Categorical Columns', ha='center', va='center')
        axes[1, 0].set_title('Categorical Distribution - None')
    
    # 5. Correlation heatmap (numerical columns only)
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[1, 1], fmt='.2f')
        axes[1, 1].set_title('Correlation Matrix')
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient Numerical Columns', ha='center', va='center')
        axes[1, 1].set_title('Correlation Matrix - N/A')
    
    # 6. Data types summary
    dtype_counts = df.dtypes.value_counts()
    axes[1, 2].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
    axes[1, 2].set_title('Data Types Distribution')
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Data distribution plot saved to {save_path}")
    
    plt.show()


def plot_model_comparison(model_results: Dict[str, Dict[str, Any]],
                         metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
                         title: str = "Model Performance Comparison",
                         figsize: Tuple[int, int] = (15, 10),
                         save_path: Optional[str] = None,
                         logger: Optional[Any] = None) -> None:
    """
    Plot model performance comparison
    
    Args:
        model_results: Dictionary of model results
        metrics: List of metrics to compare
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
    """
    if not model_results:
        if logger:
            logger.warning("No model results available")
        return
    
    # Extract data for comparison
    comparison_data = []
    for model_name, results in model_results.items():
        if 'error' not in results:
            test_metrics = results.get('test_metrics', {})
            row = {'Model': model_name}
            for metric in metrics:
                row[metric] = test_metrics.get(metric, 0)
            comparison_data.append(row)
    
    if not comparison_data:
        if logger:
            logger.warning("No valid model results for comparison")
        return
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Create subplots
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
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
            sns.barplot(data=df_comparison, x='Model', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_ylim(0, 1)  # Assuming metrics are between 0 and 1
    
    # Hide unused subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         labels: Optional[List[str]] = None,
                         title: str = "Confusion Matrix",
                         figsize: Tuple[int, int] = (8, 6),
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save plot if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray,
                   title: str = "ROC Curve",
                   figsize: Tuple[int, int] = (8, 6),
                   save_path: Optional[str] = None) -> None:
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    
    if y_proba.shape[1] > 1:
        y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = y_proba.flatten()
    
    fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
    auc_score = roc_auc_score(y_true, y_proba_pos)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve plot saved to {save_path}")
    
    plt.show()


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray,
                               title: str = "Precision-Recall Curve",
                               figsize: Tuple[int, int] = (8, 6),
                               save_path: Optional[str] = None) -> None:
    """
    Plot Precision-Recall curve
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    if y_proba.shape[1] > 1:
        y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = y_proba.flatten()
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba_pos)
    ap_score = average_precision_score(y_true, y_proba_pos)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {ap_score:.2f})')
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve plot saved to {save_path}")
    
    plt.show()


def create_dashboard(model_results: Dict[str, Dict[str, Any]],
                    data_info: Dict[str, Any],
                    save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive dashboard
    
    Args:
        model_results: Dictionary of model results
        data_info: Data information dictionary
        save_path: Path to save the dashboard
    """
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('ML1 Loan Prediction - Comprehensive Dashboard', fontsize=20, fontweight='bold')
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Model Performance Comparison (top row, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    if model_results:
        comparison_data = []
        for model_name, results in model_results.items():
            if 'error' not in results:
                test_metrics = results.get('test_metrics', {})
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': test_metrics.get('accuracy', 0),
                    'F1-Score': test_metrics.get('f1_score', 0)
                })
        
        if comparison_data:
            df_comp = pd.DataFrame(comparison_data)
            x = np.arange(len(df_comp))
            width = 0.35
            
            ax1.bar(x - width/2, df_comp['Accuracy'], width, label='Accuracy', alpha=0.8)
            ax1.bar(x + width/2, df_comp['F1-Score'], width, label='F1-Score', alpha=0.8)
            
            ax1.set_xlabel('Models')
            ax1.set_ylabel('Score')
            ax1.set_title('Model Performance Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(df_comp['Model'], rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
    
    # 2. Data Overview (top right)
    ax2 = fig.add_subplot(gs[0, 2:])
    if data_info:
        info_text = f"""
        Dataset Shape: {data_info.get('shape', 'N/A')}
        Features: {data_info.get('n_features', 'N/A')}
        Samples: {data_info.get('n_samples', 'N/A')}
        Missing Values: {data_info.get('missing_count', 'N/A')}
        """
        ax2.text(0.1, 0.5, info_text, transform=ax2.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax2.set_title('Dataset Overview')
        ax2.axis('off')
    
    # 3. Feature Importance (middle row, left)
    ax3 = fig.add_subplot(gs[1, :2])
    if model_results:
        # Get feature importance from the best model
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
                
                ax3.barh(range(len(features)), importances, color='skyblue', alpha=0.7)
                ax3.set_yticks(range(len(features)))
                ax3.set_yticklabels(features)
                ax3.set_xlabel('Importance')
                ax3.set_title('Top 10 Feature Importance')
                ax3.invert_yaxis()
    
    # 4. Model Metrics Summary (middle row, right)
    ax4 = fig.add_subplot(gs[1, 2:])
    if model_results:
        metrics_data = []
        for model_name, results in model_results.items():
            if 'error' not in results:
                test_metrics = results.get('test_metrics', {})
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': test_metrics.get('accuracy', 0),
                    'Precision': test_metrics.get('precision', 0),
                    'Recall': test_metrics.get('recall', 0),
                    'F1-Score': test_metrics.get('f1_score', 0)
                })
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            # Create a heatmap
            metrics_matrix = df_metrics.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
            sns.heatmap(metrics_matrix, annot=True, cmap='YlOrRd', ax=ax4, fmt='.3f')
            ax4.set_title('Model Metrics Heatmap')
    
    # 5. Training Summary (bottom row)
    ax5 = fig.add_subplot(gs[2, :])
    summary_text = f"""
    ML1 Loan Prediction System - Training Summary
    
    • Total Models Trained: {len([r for r in model_results.values() if 'error' not in r])}
    • Best Model: {max(model_results.items(), key=lambda x: x[1].get('test_metrics', {}).get('accuracy', 0))[0] if model_results else 'N/A'}
    • Best Accuracy: {max([r.get('test_metrics', {}).get('accuracy', 0) for r in model_results.values() if 'error' not in r]):.4f if model_results else 0}
    • Training Completed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    ax5.text(0.1, 0.5, summary_text, transform=ax5.transAxes, fontsize=14,
            verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
    ax5.set_title('Training Summary')
    ax5.axis('off')
    
    # Save dashboard if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")
    
    plt.show()
