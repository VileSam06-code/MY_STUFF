"""
UPI Fraud Detection - Data Loading and Preprocessing Module

This module loads and preprocesses the UPI fraud dataset with features:
- trans_hour, trans_day, trans_month, trans_year: Temporal features
- category: Merchant category code
- upi_number: UPI transaction ID
- age: Customer age
- trans_amount: Transaction amount (numerical)
- state, zip: Geographic location features
- fraud_risk: Target variable (0=legitimate, 1=fraud)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Load and preprocess UPI fraud detection dataset.
    """
    
    def __init__(self, dataset_path: str = 'upi_fraud_dataset.csv'):
        self.dataset_path = dataset_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = RobustScaler()
        self.feature_cols = None
        self.target_col = 'fraud_risk'
        
    def load_data(self):
        """
        Load the UPI fraud dataset.
        Handles both root and data/ folder locations.
        """
        try:
            # Try multiple paths for flexibility
            possible_paths = [
                self.dataset_path,
                os.path.join('data', self.dataset_path),
                os.path.join('..', self.dataset_path),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    logger.info(f"Loading dataset from: {path}")
                    self.df = pd.read_csv(path)
                    break
            
            if self.df is None:
                raise FileNotFoundError(f"Dataset not found in any of the expected locations")
            
            logger.info(f"Dataset shape: {self.df.shape}")
            logger.info(f"Columns: {self.df.columns.tolist()}")
            logger.info(f"Missing values:\n{self.df.isnull().sum()}")
            logger.info(f"Fraud distribution:\n{self.df[self.target_col].value_counts()}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess(self, test_size=0.2, random_state=42, apply_smote=True):
        """
        Preprocess the dataset for model training.
        """
        if self.df is None:
            self.load_data()
        
        # Define feature columns (exclude id and target)
        feature_cols = [col for col in self.df.columns 
                       if col not in ['id', 'upi_number', self.target_col]]
        self.feature_cols = feature_cols
        
        X = self.df[feature_cols].copy()
        y = self.df[self.target_col].copy()
        
        logger.info(f"Features selected: {feature_cols}")
        logger.info(f"Feature space shape: {X.shape}")
        
        # Handle missing values
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0, inplace=True)
        
        logger.info("Missing values handled")
        
        # Feature scaling
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=feature_cols,
            index=X.index
        )
        
        # Train-test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Train set size: {self.X_train.shape[0]}")
        logger.info(f"Test set size: {self.X_test.shape[0]}")
        logger.info(f"Train fraud rate: {self.y_train.mean():.4f}")
        logger.info(f"Test fraud rate: {self.y_test.mean():.4f}")
        
        # Apply SMOTE if requested
        if apply_smote:
            logger.info("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=random_state)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            logger.info(f"After SMOTE - Train shape: {self.X_train.shape}")
            logger.info(f"After SMOTE - Fraud rate: {self.y_train.mean():.4f}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_class_weights(self):
        """
        Compute class weights for imbalanced classification.
        """
        if self.y_train is None:
            raise ValueError("Data not preprocessed yet. Call preprocess() first.")
        
        classes = np.unique(self.y_train)
        weights = compute_class_weight('balanced', classes=classes, y=self.y_train)
        class_weight_dict = {cls: weight for cls, weight in zip(classes, weights)}
        
        logger.info(f"Class weights: {class_weight_dict}")
        return class_weight_dict
    
    def get_feature_info(self):
        """
        Get information about loaded features.
        """
        if self.feature_cols is None:
            raise ValueError("Data not preprocessed yet. Call preprocess() first.")
        
        return {
            'feature_columns': self.feature_cols,
            'n_features': len(self.feature_cols),
            'target_column': self.target_col,
            'n_samples_train': len(self.X_train),
            'n_samples_test': len(self.X_test)
        }


def load_and_preprocess_data(filepath='upi_fraud_dataset.csv', test_size=0.2, random_state=42, apply_smote=True):
    """
    Convenience function to load and preprocess data in one call.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file (relative or absolute)
    test_size : float
        Test set proportion
    random_state : int
        Random seed
    apply_smote : bool
        Whether to apply SMOTE balancing
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : array-like
        Preprocessed train-test split data
    """
    loader = DataLoader(filepath)
    loader.load_data()
    return loader.preprocess(test_size, random_state, apply_smote)
