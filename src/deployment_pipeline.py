"""
End-to-End Fraud Detection Deployment Pipeline

This module orchestrates the complete ML pipeline for UPI fraud detection:
1. Data loading and preprocessing
2. Feature engineering
3. Model training (XGBoost, Deep Learning, Ensemble)
4. Model evaluation and comparison
5. Inference and predictions
6. Model persistence
"""

import numpy as np
import pandas as pd
import pickle
import joblib
from datetime import datetime
import logging
from typing import Dict, Tuple, Any, List

from data_loader import DataLoader, load_and_preprocess_data
from xgboost_model import XGBoostFraudDetector
from deep_learning_model import DeepLearningDetector
from ensemble_model import EnsembleDetector
from model_evaluation import ModelEvaluator
from feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionPipeline:
    """
    Complete end-to-end fraud detection pipeline for UPI transactions.
    """
    
    def __init__(self, dataset_path: str = 'upi_fraud_dataset.csv'):
        """
        Initialize the fraud detection pipeline.
        
        Parameters:
        -----------
        dataset_path : str
            Path to the UPI fraud dataset CSV file
        """
        self.dataset_path = dataset_path
        self.data_loader = DataLoader(dataset_path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.models = {}
        self.results = {}
        self.scaler = None
        
        logger.info("UPI Fraud Detection Pipeline initialized")
    
    def load_and_preprocess(self, test_size=0.2, apply_smote=True) -> Tuple:
        """
        Load and preprocess the UPI fraud dataset.
        """
        logger.info("Starting data loading and preprocessing...")
        self.data_loader.load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.data_loader.preprocess(test_size=test_size, apply_smote=apply_smote)
        
        logger.info(f"Train samples: {self.X_train.shape[0]}, Test samples: {self.X_test.shape[0]}")
        logger.info(f"Train fraud rate: {self.y_train.mean():.4f}")
        logger.info(f"Test fraud rate: {self.y_test.mean():.4f}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self) -> Dict:
        """
        Train all three types of models: XGBoost, Deep Learning, and Ensemble.
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING XGBoost MODEL")
        logger.info("="*60)
        self.models['xgboost'] = XGBoostFraudDetector()
        self.models['xgboost'].train(self.X_train, self.y_train)
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING DEEP LEARNING MODEL")
        logger.info("="*60)
        self.models['deep_learning'] = DeepLearningDetector()
        self.models['deep_learning'].train(self.X_train, self.y_train)
        
        logger.info("\n" + "="*60)
        logger.info("CREATING ENSEMBLE MODEL")
        logger.info("="*60)
        self.models['ensemble'] = EnsembleDetector([
            self.models['xgboost'],
            self.models['deep_learning']
        ])
        self.models['ensemble'].train(self.X_train, self.y_train)
        
        logger.info("\nAll models trained successfully!")
        return self.models
    
    def evaluate_models(self) -> Dict:
        """
        Evaluate all trained models on test set.
        """
        if not self.models:
            raise ValueError("Models not trained. Call train_models() first.")
        
        logger.info("\n" + "="*60)
        logger.info("MODEL EVALUATION AND COMPARISON")
        logger.info("="*60)
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate(
            [self.models['xgboost'], 
             self.models['deep_learning'], 
             self.models['ensemble']],
            self.X_test,
            self.y_test
        )
        
        self.results = results
        return results
    
    def predict(self, X_new: pd.DataFrame, model_name: str = 'ensemble') -> np.ndarray:
        """
        Make predictions on new data using specified model.
        
        Parameters:
        -----------
        X_new : pd.DataFrame
            New feature data for prediction
        model_name : str
            Name of the model to use ('xgboost', 'deep_learning', 'ensemble')
        
        Returns:
        --------
        np.ndarray : Fraud predictions (0/1)
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        return self.models[model_name].predict(X_new)
    
    def predict_proba(self, X_new: pd.DataFrame, model_name: str = 'ensemble') -> np.ndarray:
        """
        Get fraud probability predictions on new data.
        
        Parameters:
        -----------
        X_new : pd.DataFrame
            New feature data for prediction
        model_name : str
            Name of the model to use
        
        Returns:
        --------
        np.ndarray : Fraud probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        return self.models[model_name].predict_proba(X_new)
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to save
        filepath : str
            Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        joblib.dump(self.models[model_name], filepath)
        logger.info(f"Model '{model_name}' saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str) -> None:
        """
        Load a saved model from disk.
        
        Parameters:
        -----------
        model_name : str
            Name to assign to the loaded model
        filepath : str
            Path to the saved model file
        """
        self.models[model_name] = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def get_summary(self) -> Dict:
        """
        Get summary of pipeline execution.
        """
        return {
            'models_trained': list(self.models.keys()),
            'n_train_samples': self.X_train.shape[0] if self.X_train is not None else None,
            'n_test_samples': self.X_test.shape[0] if self.X_test is not None else None,
            'train_fraud_rate': self.y_train.mean() if self.y_train is not None else None,
            'test_fraud_rate': self.y_test.mean() if self.y_test is not None else None,
            'evaluation_results': self.results
        }


def run_complete_pipeline(dataset_path: str = 'upi_fraud_dataset.csv') -> FraudDetectionPipeline:
    """
    Execute the complete UPI fraud detection pipeline.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the UPI fraud dataset
    
    Returns:
    --------
    FraudDetectionPipeline : Trained and evaluated pipeline object
    """
    logger.info("\n" + "#"*60)
    logger.info("# UPI FRAUD DETECTION - COMPLETE PIPELINE")
    logger.info("#"*60 + "\n")
    
    # Initialize pipeline
    pipeline = FraudDetectionPipeline(dataset_path)
    
    # Load and preprocess data
    pipeline.load_and_preprocess(test_size=0.2, apply_smote=True)
    
    # Train models
    pipeline.train_models()
    
    # Evaluate models
    pipeline.evaluate_models()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*60)
    summary = pipeline.get_summary()
    for key, value in summary.items():
        logger.info(f"{key}: {value}")
    
    return pipeline


if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = run_complete_pipeline()
    
    # Example: Make predictions on test set
    logger.info("\nMaking predictions on test set...")
    predictions = pipeline.predict(pipeline.X_test, model_name='ensemble')
    probabilities = pipeline.predict_proba(pipeline.X_test, model_name='ensemble')
    
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Fraud rate in predictions: {predictions.mean():.4f}")
