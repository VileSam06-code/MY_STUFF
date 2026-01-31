"""
UPI Fraud Detection - Feature Engineering Module

This module implements transaction-level feature engineering for UPI fraud detection.
Specifically designed for the UPI fraud dataset with the following characteristics:
- Transaction-level data (no customer tracking)
- 2665 samples with 9:1 class imbalance
- Features: temporal (hour, day, month), amount, category, age, geographic (state, zip)

Features Implemented:
1. Cyclic Temporal Encoding: sin/cos encoding for hour, day, month
2. Amount-based Features: log-transform, standardization, percentile ranks
3. Categorical Encoding: one-hot and frequency encoding
4. Geographic Features: state binning and zip code grouping
5. Interaction Features: amount-category, amount-age interactions
6. Statistical Features: amount deviation scores
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from typing import Tuple, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UPIFraudFeatureEngineer:
    """
    Transaction-level feature engineering for UPI fraud detection.
    Optimized for 2665-sample dataset without customer-level aggregations.
    """
    
    def __init__(self):
        """
        Initialize the feature engineer.
        """
        self.numeric_features = ['age', 'trans_amount']
        self.categorical_features = ['category', 'state']
        self.temporal_features = ['trans_hour', 'trans_day', 'trans_month']
        self.scaler = StandardScaler()
        self.encoder = None
        self.feature_names = []
        
        logger.info("UPI Fraud Feature Engineer initialized")
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform features for training data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features (should exclude 'id', 'upi_number', 'fraud_risk')
        
        Returns:
        --------
        pd.DataFrame : Transformed feature matrix
        """
        logger.info(f"Starting feature engineering on {X.shape[0]} samples")
        
        X_processed = X.copy()
        
        # Step 1: Cyclic Temporal Encoding
        logger.info("Encoding temporal features...")
        X_processed = self._encode_temporal_features(X_processed)
        
        # Step 2: Amount-based Features
        logger.info("Engineering amount-based features...")
        X_processed = self._engineer_amount_features(X_processed)
        
        # Step 3: One-hot encode categorical variables
        logger.info("Encoding categorical features...")
        X_processed = self._encode_categorical_features(X_processed, fit=True)
        
        # Step 4: Geographic feature engineering
        logger.info("Engineering geographic features...")
        X_processed = self._engineer_geographic_features(X_processed)
        
        # Step 5: Interaction features
        logger.info("Creating interaction features...")
        X_processed = self._create_interaction_features(X_processed)
        
        # Step 6: Scale numeric features
        logger.info("Scaling features...")
        X_processed = self._scale_features(X_processed, fit=True)
        
        logger.info(f"Feature engineering complete. Output shape: {X_processed.shape}")
        return X_processed
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test/new data using fitted feature engineer.
        """
        X_processed = X.copy()
        X_processed = self._encode_temporal_features(X_processed)
        X_processed = self._engineer_amount_features(X_processed)
        X_processed = self._encode_categorical_features(X_processed, fit=False)
        X_processed = self._engineer_geographic_features(X_processed)
        X_processed = self._create_interaction_features(X_processed)
        X_processed = self._scale_features(X_processed, fit=False)
        return X_processed
    
    def _encode_temporal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encode temporal features using cyclic encoding (sin/cos).
        Accounts for cyclical nature of time.
        """
        X = X.copy()
        
        # Hour (0-23): 24-hour cycle
        X['trans_hour_sin'] = np.sin(2 * np.pi * X['trans_hour'] / 24)
        X['trans_hour_cos'] = np.cos(2 * np.pi * X['trans_hour'] / 24)
        
        # Day (1-31): 31-day cycle
        X['trans_day_sin'] = np.sin(2 * np.pi * X['trans_day'] / 31)
        X['trans_day_cos'] = np.cos(2 * np.pi * X['trans_day'] / 31)
        
        # Month (1-12): 12-month cycle
        X['trans_month_sin'] = np.sin(2 * np.pi * X['trans_month'] / 12)
        X['trans_month_cos'] = np.cos(2 * np.pi * X['trans_month'] / 12)
        
        # Drop original temporal columns
        X = X.drop(['trans_hour', 'trans_day', 'trans_month', 'trans_year'], axis=1)
        
        return X
    
    def _engineer_amount_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features based on transaction amount.
        """
        X = X.copy()
        
        # Log transform (handle zero values)
        X['amount_log'] = np.log1p(X['trans_amount'])
        
        # Percentile rank (captures relative position)
        X['amount_percentile'] = X['trans_amount'].rank(pct=True)
        
        # Statistical deviation (Z-score like)
        amount_mean = X['trans_amount'].mean()
        amount_std = X['trans_amount'].std()
        X['amount_zscore'] = (X['trans_amount'] - amount_mean) / (amount_std + 1e-8)
        
        # Amount bins (categorical)
        X['amount_bin'] = pd.qcut(X['trans_amount'], q=5, labels=False, duplicates='drop')
        
        return X
    
    def _encode_categorical_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        One-hot encode categorical variables.
        """
        X = X.copy()
        
        # One-hot encode category and state
        if fit:
            self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            categorical_encoded = self.encoder.fit_transform(X[['category', 'state']])
            categorical_cols = self.encoder.get_feature_names_out(['category', 'state'])
        else:
            categorical_encoded = self.encoder.transform(X[['category', 'state']])
            categorical_cols = self.encoder.get_feature_names_out(['category', 'state'])
        
        # Create dataframe with encoded features
        X_categorical = pd.DataFrame(categorical_encoded, columns=categorical_cols, index=X.index)
        
        # Drop original categorical columns and zip (will process separately)
        X = X.drop(['category', 'state'], axis=1)
        X = pd.concat([X, X_categorical], axis=1)
        
        return X
    
    def _engineer_geographic_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer geographic features (zip code clustering).
        """
        X = X.copy()
        
        # Zip code binning (group into regions)
        X['zip_region'] = pd.cut(X['zip'], bins=5, labels=False)
        X = X.drop(['zip'], axis=1)
        
        return X
    
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between amount and other numeric variables.
        """
        X = X.copy()
        
        # Amount * Age interaction
        X['amount_age_interaction'] = X['trans_amount'] * X['age']
        
        # Amount * amount_percentile
        if 'amount_percentile' in X.columns:
            X['amount_intensity'] = X['trans_amount'] * X['amount_percentile']
        
        # Age-based features
        X['age_squared'] = X['age'] ** 2
        X['age_log'] = np.log1p(X['age'])
        
        return X
    
    def _scale_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Scale numeric features using StandardScaler.
        """
        X = X.copy()
        
        # Identify numeric columns (excluding one-hot encoded)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        else:
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        
        self.feature_names = X.columns.tolist()
        return X
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of engineered feature names.
        """
        return self.feature_names
    
    def get_feature_count(self) -> int:
        """
        Get count of engineered features.
        """
        return len(self.feature_names) if self.feature_names else 0


# Convenience function for compatibility
def create_feature_engineer():
    """
    Factory function to create feature engineer instance.
    """
    return UPIFraudFeatureEngineer()
