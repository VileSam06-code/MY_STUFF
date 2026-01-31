"""
Fraud Detection Configuration

Centralized configuration for all models, data paths, and hyperparameters.
Modify these settings for your specific use case.
"""

import os
from pathlib import Path

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / 'data'
DATASET_PATH = DATA_DIR / 'upi_fraud_dataset.csv'

# Dataset parameters
DATA_CONFIG = {
    'filepath': str(DATASET_PATH),
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'apply_smote': True,  # Apply SMOTE for class balancing in training
    'target_column': 'fraud_risk',
}

# ============================================================================
# XGBOOST MODEL CONFIGURATION
# ============================================================================

XGBOOST_CONFIG = {
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 200,
    'objective': 'binary:logistic',
    'random_state': 42,
    'eval_metric': 'auc',
    'early_stopping_rounds': 20,
    'scale_pos_weight': 'auto',  # Automatically calculate based on class imbalance
    'gpu_id': 0,  # Set to -1 to use CPU
    'tree_method': 'hist',  # Use 'gpu_hist' for GPU acceleration
}

# ============================================================================
# DEEP LEARNING MODEL CONFIGURATION
# ============================================================================

DEEP_LEARNING_CONFIG = {
    'hidden_units': [128, 64, 32],
    'dropout_rate': 0.3,
    'batch_normalization': True,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss_function': 'binary_crossentropy',
    'epochs': 50,
    'batch_size': 32,
    'validation_split': 0.1,
    'early_stopping_patience': 10,
    'random_state': 42,
    'metrics': ['auc', 'precision', 'recall', 'f1'],
}

# ============================================================================
# ENSEMBLE MODEL CONFIGURATION
# ============================================================================

ENSEMBLE_CONFIG = {
    'method': 'stacking',  # Options: 'stacking', 'voting', 'blending'
    'base_models': ['xgboost', 'deep_learning'],
    'meta_learner': 'logistic_regression',  # Options: 'logistic_regression', 'xgboost'
    'stack_on_test': True,
    'random_state': 42,
}

# ============================================================================
# MODEL EVALUATION CONFIGURATION
# ============================================================================

EVALUATION_CONFIG = {
    'metrics': [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'roc_auc',
        'pr_auc',
        'confusion_matrix',
        'classification_report',
    ],
    'threshold_candidates': [0.3, 0.4, 0.5, 0.6, 0.7],  # For threshold optimization
    'compute_shap': False,  # Set to True to compute SHAP values (computationally expensive)
}

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================

FEATURE_ENGINEERING_CONFIG = {
    'apply_temporal_features': True,
    'apply_amount_statistics': True,
    'apply_geographic_features': True,
    'apply_interaction_features': True,
    'apply_velocity_features': True,
    'apply_categorical_encoding': True,
    'categorical_encoding_method': 'target_encoding',  # Options: 'target_encoding', 'onehot'
    'scaling_method': 'robust',  # Options: 'standard', 'robust', 'minmax'
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    'n_jobs': -1,  # Number of parallel jobs (-1 = use all cores)
    'verbose': True,
    'save_models': True,
    'models_dir': Path(__file__).parent.parent / 'models',
    'logs_dir': Path(__file__).parent.parent / 'logs',
}

# Create directories if they don't exist
TRAINING_CONFIG['models_dir'].mkdir(parents=True, exist_ok=True)
TRAINING_CONFIG['logs_dir'].mkdir(parents=True, exist_ok=True)

# ============================================================================
# PREDICTION CONFIGURATION
# ============================================================================

PREDICTION_CONFIG = {
    'default_threshold': 0.5,  # Default classification threshold
    'return_probabilities': True,
    'model_type': 'ensemble',  # Options: 'xgboost', 'deep_learning', 'ensemble'
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': str(TRAINING_CONFIG['logs_dir'] / 'fraud_detection.log'),
}

# ============================================================================
# MODEL PATHS
# ============================================================================

MODEL_PATHS = {
    'xgboost': str(TRAINING_CONFIG['models_dir'] / 'xgboost_detector.pkl'),
    'deep_learning': str(TRAINING_CONFIG['models_dir'] / 'deep_learning_detector.h5'),
    'ensemble': str(TRAINING_CONFIG['models_dir'] / 'ensemble_detector.pkl'),
    'feature_scaler': str(TRAINING_CONFIG['models_dir'] / 'feature_scaler.pkl'),
}
