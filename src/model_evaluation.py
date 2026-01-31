"""
Model Evaluation Module

Comprehensive evaluation framework for fraud detection models including:
- Standard metrics (ROC-AUC, F1, Precision, Recall)
- Business metrics (cost-benefit analysis)
- Comparative analysis
- Threshold optimization
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, confusion_matrix,
    classification_report, auc
)
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison framework.
    """
    
    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray = None,
                      threshold: float = 0.5) -> Dict:
        """
        Evaluate model on standard metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_proba (np.ndarray): Predicted probabilities
            y_pred (np.ndarray): Binary predictions (optional)
            threshold (float): Classification threshold
            
        Returns:
            dict: Evaluation metrics
        """
        if y_pred is None:
            y_pred = (y_proba > threshold).astype(int)
        
        metrics = {
            'roc_auc': roc_auc_score(y_true, y_proba),
            'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'tn': confusion_matrix(y_true, y_pred)[0, 0],
            'fp': confusion_matrix(y_true, y_pred)[0, 1],
            'fn': confusion_matrix(y_true, y_pred)[1, 0],
            'tp': confusion_matrix(y_true, y_pred)[1, 1],
        }
        
        return metrics
    
    @staticmethod
    def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                          metric: str = 'f1') -> Tuple[float, float]:
        """
        Find optimal classification threshold.
        
        Args:
            y_true (np.ndarray): True labels
            y_proba (np.ndarray): Predicted probabilities
            metric (str): Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            Tuple[float, float]: Optimal threshold and metric value
        """
        thresholds = np.arange(0.1, 1.0, 0.01)
        scores = []
        
        for thresh in thresholds:
            y_pred = (y_proba > thresh).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)
            
            scores.append(score)
        
        best_idx = np.argmax(scores)
        best_threshold = thresholds[best_idx]
        best_score = scores[best_idx]
        
        logger.info(f"Optimal threshold for {metric}: {best_threshold:.3f} (score: {best_score:.4f})")
        return best_threshold, best_score
    
    @staticmethod
    def compare_models(models_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            models_results (Dict): Dictionary mapping model names to their metrics
            
        Returns:
            pd.DataFrame: Comparison table
        """
        df = pd.DataFrame(models_results).T
        df = df.sort_values('roc_auc', ascending=False)
        
        logger.info(f"\nModel Comparison:\n{df.to_string()}")
        return df
    
    @staticmethod
    def business_metrics(y_true: np.ndarray, y_proba: np.ndarray,
                        false_positive_cost: float = 1.0,
                        false_negative_cost: float = 10.0,
                        threshold: float = 0.5) -> Dict:
        """
        Calculate business-relevant metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_proba (np.ndarray): Predicted probabilities
            false_positive_cost (float): Cost of false positive
            false_negative_cost (float): Cost of false negative
            threshold (float): Classification threshold
            
        Returns:
            dict: Business metrics
        """
        y_pred = (y_proba > threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        
        total_cost = fp * false_positive_cost + fn * false_negative_cost
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics = {
            'total_cost': total_cost,
            'detection_rate': detection_rate,
            'false_alarm_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'cost_per_fraud': total_cost / (tp + fn) if (tp + fn) > 0 else 0,
        }
        
        return metrics


if __name__ == "__main__":
    print("Model Evaluation Module")
