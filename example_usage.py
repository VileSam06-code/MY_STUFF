"""
Example Usage of Fraud Detection Pipeline

This script demonstrates how to use the fraud detection package step-by-step.
Students can run this to understand the complete workflow and train models.

Run with: python example_usage.py
"""

import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
SRC_PATH = Path(__file__).parent / 'src'
sys.path.insert(0, str(SRC_PATH))

def example_1_load_and_preprocess_data():
    """
    Example 1: Load and preprocess UPI fraud dataset
    """
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 1: Load and Preprocess Data")
    logger.info("="*80)
    
    try:
        from src.data_loader import DataLoader
        from src.config import DATA_CONFIG
        
        logger.info("Initializing DataLoader...")
        loader = DataLoader(DATA_CONFIG['filepath'])
        
        logger.info("Loading dataset...")
        df = loader.load_data()
        logger.info(f"Dataset shape: {df.shape}")
        
        logger.info("Preprocessing dataset...")
        X_train, X_test, y_train, y_test = loader.preprocess(
            test_size=DATA_CONFIG['test_size'],
            random_state=DATA_CONFIG['random_state'],
            apply_smote=DATA_CONFIG['apply_smote']
        )
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        logger.info(f"Training fraud rate: {y_train.mean():.4f}")
        logger.info(f"Test fraud rate: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test, loader
        
    except Exception as e:
        logger.error(f"Error in Example 1: {str(e)}", exc_info=True)
        logger.info("TIP: Make sure you have the dataset in data/upi_fraud_dataset.csv")
        return None, None, None, None, None

def example_2_train_xgboost(X_train, X_test, y_train, y_test):
    """
    Example 2: Train XGBoost model
    """
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 2: Train XGBoost Model")
    logger.info("="*80)
    
    try:
        from src.xgboost_model import XGBoostFraudDetector
        from src.config import XGBOOST_CONFIG
        
        logger.info("Initializing XGBoost detector...")
        xgb = XGBoostFraudDetector(
            learning_rate=XGBOOST_CONFIG['learning_rate'],
            max_depth=XGBOOST_CONFIG['max_depth'],
            n_estimators=XGBOOST_CONFIG['n_estimators']
        )
        
        logger.info("Training XGBoost model...")
        xgb.train(X_train, y_train, X_test, y_test)
        
        logger.info("Making predictions...")
        y_pred = xgb.predict(X_test)
        y_pred_proba = xgb.predict_proba(X_test)
        
        logger.info(f"Predictions shape: {y_pred.shape}")
        logger.info(f"Prediction probabilities shape: {y_pred_proba.shape}")
        logger.info(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        return xgb, y_pred, y_pred_proba
        
    except Exception as e:
        logger.error(f"Error in Example 2: {str(e)}", exc_info=True)
        return None, None, None

def example_3_train_deep_learning(X_train, X_test, y_train, y_test):
    """
    Example 3: Train Deep Learning model
    """
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 3: Train Deep Learning Model")
    logger.info("="*80)
    
    try:
        from src.deep_learning_model import DeepLearningDetector
        from src.config import DEEP_LEARNING_CONFIG
        
        logger.info("Initializing Deep Learning detector...")
        dl = DeepLearningDetector(
            input_dim=X_train.shape[1],
            hidden_units=DEEP_LEARNING_CONFIG['hidden_units'],
            dropout_rate=DEEP_LEARNING_CONFIG['dropout_rate']
        )
        
        logger.info("Building model...")
        dl.build_model()
        
        logger.info("Training Deep Learning model...")
        dl.train(
            X_train, y_train,
            X_test, y_test,
            epochs=DEEP_LEARNING_CONFIG['epochs']
        )
        
        logger.info("Making predictions...")
        y_pred = dl.predict(X_test)
        y_pred_proba = dl.predict_proba(X_test)
        
        logger.info(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        return dl, y_pred, y_pred_proba
        
    except Exception as e:
        logger.error(f"Error in Example 3: {str(e)}", exc_info=True)
        logger.info("TIP: Make sure TensorFlow is installed: pip install tensorflow")
        return None, None, None

def example_4_evaluate_model(y_test, y_pred, model_name="Model"):
    """
    Example 4: Evaluate model performance
    """
    logger.info("\n" + "="*80)
    logger.info(f"EXAMPLE 4: Evaluate {model_name} Performance")
    logger.info("="*80)
    
    try:
        logger.info("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\n{cm}")
        
        logger.info("\nClassification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
    except Exception as e:
        logger.error(f"Error in Example 4: {str(e)}", exc_info=True)

def main():
    """
    Main execution function
    """
    logger.info("\n")
    logger.info("*" * 80)
    logger.info("FRAUD DETECTION PIPELINE - EXAMPLE USAGE")
    logger.info("*" * 80)
    
    # Example 1: Load and preprocess data
    X_train, X_test, y_train, y_test, loader = example_1_load_and_preprocess_data()
    
    if X_train is None:
        logger.error("Failed to load data. Stopping execution.")
        return
    
    # Example 2: Train XGBoost model
    xgb, xgb_pred, xgb_proba = example_2_train_xgboost(X_train, X_test, y_train, y_test)
    if xgb is not None:
        example_4_evaluate_model(y_test, xgb_pred, "XGBoost")
    
    # Example 3: Train Deep Learning model
    # Uncomment if TensorFlow is available
    # dl, dl_pred, dl_proba = example_3_train_deep_learning(X_train, X_test, y_train, y_test)
    # if dl is not None:
    #     example_4_evaluate_model(y_test, dl_pred, "Deep Learning")
    
    logger.info("\n" + "="*80)
    logger.info("EXAMPLES COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info("\nNext Steps:")
    logger.info("1. Review the main_training.py script for the complete pipeline")
    logger.info("2. Check src/config.py to customize hyperparameters")
    logger.info("3. Examine src/ directory for detailed model implementations")
    logger.info("4. Read the README.md and INSTRUCTIONS.md for more details")
    logger.info("\n")

if __name__ == '__main__':
    main()
