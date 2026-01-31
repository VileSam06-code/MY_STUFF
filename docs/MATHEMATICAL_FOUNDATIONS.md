# Mathematical Foundations of Fraud Detection

## Classification Problem

### Binary Classification
Predicting fraud (1) vs. legitimate (0):
$$P(y=1|x) = \sigma(w^T x + b)$$

Where $\sigma$ is the sigmoid function.

## Logistic Regression Loss
$$L = -\frac{1}{n}\sum_i [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

## Gradient Descent
$$w := w - \eta \nabla L(w)$$

Where $\eta$ is learning rate and $\nabla L$ is gradient.

## Evaluation Metrics

### Confusion Matrix
- **TP (True Positive)**: Correctly predicted fraud
- **FP (False Positive)**: Incorrectly predicted fraud  
- **FN (False Negative)**: Missed fraud
- **TN (True Negative)**: Correct legitimate prediction

### ROC-AUC
Area under Receiver Operating Characteristic curve.

### F1-Score  
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

## Imbalanced Learning

### Class Weights
$$w_c = \frac{n}{n_c \cdot k}$$

Where $k$ is number of classes.

## Regularization

### L1 (Lasso)
$$\Omega = \lambda \sum_i |w_i|$$

### L2 (Ridge)  
$$\Omega = \frac{\lambda}{2} \sum_i w_i^2$$

## Hyperparameter Optimization

Tune hyperparameters using cross-validation to minimize generalization error.
