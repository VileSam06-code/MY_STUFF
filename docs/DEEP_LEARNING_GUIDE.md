# Deep Learning for Fraud Detection

## Neural Network Architecture

Multi-layer perceptron (MLP) with:  
- Input Layer: Feature vector (n_features)
- Hidden Layers: [128, 64, 32] neurons with ReLU activation
- Dropout: 0.3 at each hidden layer
- Output Layer: 1 neuron with sigmoid activation

## Mathematical Foundation

### ReLU Activation
$$\text{ReLU}(x) = \max(0, x)$$

### Sigmoid Activation
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

### Binary Cross-Entropy Loss
$$L = -\frac{1}{n}\sum_i [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

### Dropout Regularization
Randomly deactivate $p\%$ of neurons during training to prevent overfitting.

## Training Strategy

- **Optimizer**: Adam (adaptive learning rate)
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Validation Split**: 20%
- **Class Weights**: Auto-calculated for imbalance

## Advantages
- Captures non-linear patterns
- Handles high-dimensional data
- Flexible architecture
- Good for feature interaction

## Implementation
See `src/deep_learning_model.py` for complete code.
