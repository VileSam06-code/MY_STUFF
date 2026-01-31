"""
Visualization and Interpretation Module

Provides visualization functions for fraud detection model analysis:
- ROC-AUC curves
- Feature importance plots
- Confusion matrix heatmaps
- Model comparison visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelVisualizer:
    """
    Visualization utilities for fraud detection models.
    """
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str = "Model",
                      save_path: str = None):
        """
        Plot ROC-AUC curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_proba (np.ndarray): Predicted probabilities
            model_name (str): Model name for plot title
            save_path (str): Path to save figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC-AUC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        logger.info(f"ROC-AUC curve saved for {model_name}")
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model",
                            save_path: str = None):
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            model_name (str): Model name for plot title
            save_path (str): Path to save figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{model_name} - Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        logger.info(f"Confusion matrix saved for {model_name}")
    
    @staticmethod
    def plot_feature_importance(feature_importance: Dict, model_name: str = "Model",
                               top_n: int = 15, save_path: str = None):
        """
        Plot feature importance.
        
        Args:
            feature_importance (Dict): Feature importance scores
            model_name (str): Model name
            top_n (int): Number of top features to display
            save_path (str): Path to save figure
        """
        if not feature_importance:
            logger.warning("No feature importance data available")
            return
        
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, scores = zip(*sorted_features)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), scores, color='steelblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(f'{model_name} - Top {top_n} Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        logger.info(f"Feature importance plot saved for {model_name}")
    
    @staticmethod
    def plot_model_comparison(results: Dict[str, Dict], metric: str = 'roc_auc',
                            save_path: str = None):
        """
        Plot model comparison.
        
        Args:
            results (Dict): Dictionary of model results
            metric (str): Metric to compare
            save_path (str): Path to save figure
        """
        models = list(results.keys())
        values = [results[m].get(metric, 0) for m in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models)])
        plt.ylabel(metric.upper())
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.ylim([0, 1.0])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        logger.info(f"Model comparison plot saved")


if __name__ == "__main__":
    print("Visualization Module")
