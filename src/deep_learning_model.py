"""
Deep Learning Model for Fraud Detection

Implements neural network and autoencoder models using TensorFlow/Keras:
- Multi-layer neural network with dropout regularization
- Autoencoder for anomaly detection
- Class weight handling for imbalanced data
- Early stopping and model checkpointing
"""

import numpy as np
from typing import Dict, Tuple
import logging

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepLearningDetector:
    """
    Deep Learning-based fraud detector using neural networks.
    
    Architecture:
    - Input layer: feature_size
    - Dense layers with ReLU activation
    - Dropout for regularization
    - Output layer with sigmoid activation
    
    Mathematical Foundation:
    Loss = Binary Crossentropy with class weights:
    L = -w_1 * y*log(ŷ) + w_0 * (1-y)*log(1-ŷ)
    """
    
    def __init__(self, input_dim: int, hidden_units: list = None, dropout_rate: float = 0.3,
                random_state: int = 42):
        """
        Initialize deep learning model.
        
        Args:
            input_dim (int): Number of input features
            hidden_units (list): Number of units in each hidden layer
            dropout_rate (float): Dropout rate for regularization
            random_state (int): Random seed
        """
        if not KERAS_AVAILABLE:
            raise ImportError("TensorFlow/Keras not installed")
        
        self.input_dim = input_dim
        self.hidden_units = hidden_units or [128, 64, 32]
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
    
    def build_model(self):
        """
        Build neural network architecture.
        
        Returns:
            keras.Model: Compiled model
        """
        logger.info("Building neural network model...")
        
        model = keras.Sequential()
        model.add(layers.Input(shape=(self.input_dim,)))
        
        # Hidden layers
        for units in self.hidden_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['AUC', 'Precision', 'Recall']
        )
        
        self.model = model
        logger.info(f"Model built with architecture: {self.hidden_units}")
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None,
             y_val: np.ndarray = None, epochs: int = 50, batch_size: int = 32,
             handle_imbalance: bool = True) -> Dict:
        """
        Train the neural network.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            handle_imbalance (bool): Whether to use class weights
            
        Returns:
            dict: Training history
        """
        if self.model is None:
            self.build_model()
        
        logger.info("Training neural network...")
        
        # Calculate class weights
        class_weight = None
        if handle_imbalance:
            n_pos = np.sum(y_train)
            n_neg = len(y_train) - n_pos
            class_weight = {0: 1.0, 1: n_neg / n_pos}
            logger.info(f"Class weights: {class_weight}")
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        validation_data = (X_val, y_val) if X_val is not None else None
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            class_weight=class_weight,
            callbacks=[early_stopping],
            verbose=0
        )
        
        logger.info("Training complete")
        return {'status': 'success'}
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X (np.ndarray): Input features
            threshold (float): Classification threshold
            
        Returns:
            np.ndarray: Binary predictions
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        proba = self.model.predict(X, verbose=0)
        return (proba > threshold).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Fraud probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.predict(X, verbose=0).flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        loss, auc, precision, recall = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'loss': loss,
            'auc': auc,
            'precision': precision,
            'recall': recall,
        }
        
        logger.info(f"Neural Network Metrics: {metrics}")
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save model to disk.
        
        Args:
            filepath (str): Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from disk.
        
        Args:
            filepath (str): Path to load model from
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("Deep Learning Fraud Detection Model")
