"""\nFraud Detection Package - Final Module

Comprehensive fraud detection implementation with:
- XGBoost gradient boosting models
- Deep learning neural networks
- Ensemble methods (stacking, voting, blending)
- Feature engineering for transaction-level data
- Model evaluation and comparison framework

Designed for engineering students learning fraud detection systems.
"""

from .xgboost_model import XGBoostFraudDetector
from .deep_learning_model import DeepLearningDetector
from .ensemble_model import EnsembleDetector
from .feature_engineering import UPIFraudFeatureEngineer
from .model_evaluation import ModelEvaluator
from .data_loader import load_and_preprocess_data
from .visualization import FraudVisualization
from .deployment_pipeline import FraudDetectionPipeline

__version__ = '1.0.0'
__author__ = 'IIT Bombay - CS Students'
__all__ = [
    'XGBoostFraudDetector',
    'DeepLearningDetector',
    'EnsembleDetector',
    'UPIFraudFeatureEngineer',
    'ModelEvaluator',
    'load_and_preprocess_data',
    'FraudVisualization',
    'FraudDetectionPipeline',
]
