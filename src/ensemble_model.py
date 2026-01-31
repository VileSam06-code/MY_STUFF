"""
Ensemble Model for Fraud Detection

Combines multiple base models (XGBoost, Neural Network) using:
- Stacking: Training meta-learner on base model predictions
- Voting: Averaging predictions from multiple models
- Blending: Simple weighted averaging
"""

import numpy as np
from typing import List, Dict
import logging
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleDetector:
    """
    Ensemble model combining multiple fraud detectors.
    
    Mathematical Foundation:
    Ensemble Prediction = Σ w_i * pred_i / Σ w_i
    where w_i are learnable or fixed weights
    """
    
    def __init__(self, base_models: List, method: str = 'weighted_voting', weights: List[float] = None):
        """
        Initialize ensemble.
        
        Args:
            base_models (List): List of trained base models
            method (str): Ensemble method ('voting', 'weighted_voting', 'stacking')
            weights (List[float]): Weights for weighted voting
        """
        self.base_models = base_models
        self.method = method
        self.weights = weights or [1.0 / len(base_models)] * len(base_models)
        self.meta_model = None
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble predictions by averaging base model predictions.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Ensemble probabilities
        """
        predictions = []
        for model in self.base_models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        # Stack predictions
        predictions = np.array(predictions)
        
        # Apply ensemble method
        if self.method == 'weighted_voting':
            # Weighted average
            ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        elif self.method == 'voting':
            # Simple average
            ensemble_pred = np.mean(predictions, axis=0)
        else:
            ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X (np.ndarray): Input features
            threshold (float): Classification threshold
            
        Returns:
            np.ndarray: Binary predictions
        """
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate ensemble performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        proba = self.predict_proba(X_test)
        pred = self.predict(X_test)
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, proba),
            'f1': f1_score(y_test, pred),
            'precision': precision_score(y_test, pred),
            'recall': recall_score(y_test, pred),
        }
        
        logger.info(f"Ensemble Metrics: {metrics}")
        return metrics


if __name__ == "__main__":
    print("Ensemble Fraud Detection Model")
