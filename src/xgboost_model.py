"""
XGBoost Model Implementation for UPI Fraud Detection

Optimized specifically for the UPI fraud dataset:
- 2665 samples with 9:1 class imbalance (9% fraud)
- Scale_pos_weight: automatic calculation for handling imbalance
- Early stopping to prevent overfitting on small dataset
- Hyperparameters tuned for fraud detection (emphasis on recall)
- Feature importance analysis
"""

import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import logging
from typing import Tuple, Optional
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostFraudDetector:
    """
    XGBoost-based fraud detection model optimized for UPI fraud dataset.
    Handles class imbalance through scale_pos_weight parameter.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize XGBoost fraud detector with UPI-optimized parameters.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.scale_pos_weight = None
        
        logger.info("XGBoost Fraud Detector initialized")
    
    def _get_upi_optimized_params(self, scale_pos_weight: float = None) -> dict:
        """
        Get XGBoost parameters optimized for UPI fraud detection.
        Accounts for 9:1 class imbalance and small dataset size (2665 samples).
        
        Parameters:
        -----------
        scale_pos_weight : float, optional
            Weight of positive class. Calculated as negative_samples / positive_samples
        
        Returns:
        --------
        dict : XGBoost parameters
        """
        # Default scale_pos_weight for 9:1 imbalance (approximately 9)
        if scale_pos_weight is None:
            scale_pos_weight = 9.0
        
        params = {
            # Basic settings
            'objective': 'binary:logistic',  # Binary classification
            'eval_metric': 'auc',  # AUC for evaluation (better for imbalanced data)
            'random_state': self.random_state,
            
            # Imbalance handling
            'scale_pos_weight': scale_pos_weight,  # Weight for positive class
            
            # Tree parameters (conservative to prevent overfitting on 2665 samples)
            'max_depth': 5,  # Shallow trees to prevent overfitting
            'min_child_weight': 3,  # Minimum sum of weights in leaf
            'subsample': 0.8,  # Use 80% of samples per tree
            'colsample_bytree': 0.8,  # Use 80% of features per tree
            'colsample_bylevel': 0.8,  # Use 80% of features per level
            
            # Regularization
            'lambda': 2.0,  # L2 regularization (prevent overfitting)
            'alpha': 0.5,  # L1 regularization
            'gamma': 0.0,  # Minimum loss reduction for split
            
            # Learning
            'learning_rate': 0.05,  # Conservative learning rate
            'n_estimators': 300,  # Number of boosting rounds (with early stopping)
            
            # Other
            'verbosity': 1,
            'n_jobs': -1  # Use all cores
        }
        
        return params
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              early_stopping_rounds: int = 20):
        """
        Train XGBoost model with early stopping.
        
        Parameters:
        -----------
        X_train : array-like or DataFrame
            Training features
        y_train : array-like
            Training labels
        X_val : array-like, optional
            Validation features for early stopping
        y_val : array-like, optional
            Validation labels for early stopping
        early_stopping_rounds : int
            Rounds without improvement before stopping
        """
        try:
            logger.info(f"Starting XGBoost training on {len(X_train)} samples")
            logger.info(f"Positive class ratio: {y_train.mean():.4f}")
            
            # Calculate scale_pos_weight
            n_negative = (y_train == 0).sum()
            n_positive = (y_train == 1).sum()
            self.scale_pos_weight = n_negative / n_positive
            
            logger.info(f"Scale pos weight: {self.scale_pos_weight:.2f}")
            
            # Get parameters
            params = self._get_upi_optimized_params(scale_pos_weight=self.scale_pos_weight)
            
            # Prepare data
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            eval_set = None
            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val)
                eval_set = [(dtrain, 'train'), (dval, 'val')]
            else:
                eval_set = [(dtrain, 'train')]
            
            # Train model
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=params['n_estimators'],
                evals=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=50
            )
            
            # Get feature importance
            self.feature_importance = self.model.get_score(importance_type='weight')
            
            logger.info("XGBoost training completed successfully")
            return self
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def predict(self, X_test, threshold: float = 0.5):
        """
        Make fraud predictions on test data.
        
        Parameters:
        -----------
        X_test : array-like or DataFrame
            Test features
        threshold : float
            Decision threshold for classification
        
        Returns:
        --------
        array : Binary predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        dtest = xgb.DMatrix(X_test)
        probabilities = self.model.predict(dtest)
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions
    
    def predict_proba(self, X_test):
        """
        Get fraud probability predictions.
        
        Parameters:
        -----------
        X_test : array-like or DataFrame
            Test features
        
        Returns:
        --------
        array : Fraud probabilities [0, 1]
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        dtest = xgb.DMatrix(X_test)
        probabilities = self.model.predict(dtest)
        
        return probabilities
    
    def get_feature_importance(self, top_n: int = 20):
        """
        Get feature importance scores.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to return
        
        Returns:
        --------
        dict : Feature importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet.")
        
        # Sort by importance
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        return dict(sorted_features)
    
    def evaluate(self, X_test, y_test, threshold: float = 0.5) -> dict:
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        threshold : float
            Decision threshold
        
        Returns:
        --------
        dict : Performance metrics
        """
        y_pred = self.predict(X_test, threshold=threshold)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            'auc_roc': roc_auc_score(y_test, y_proba),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
        }
        
        return metrics
    
    def save(self, filepath: str):
        """
        Save model to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        self.model.save_model(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to load model from
        """
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self
