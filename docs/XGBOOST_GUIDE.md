# XGBoost: Theory and Implementation for Fraud Detection

## XGBoost Algorithm Overview

XGBoost (eXtreme Gradient Boosting) is an optimized gradient boosting framework that builds an ensemble of decision trees sequentially.

## Mathematical Foundation

### Objective Function
$$\mathcal{L}(\phi) = \sum_i l(\hat{y}_i, y_i) + \sum_k \Omega(f_k)$$

Where:
- $l$ = loss function (logistic loss for binary classification)  
- $\Omega$ = regularization term = $\gamma T + \frac{1}{2}\lambda \sum_j w_j^2$
- $T$ = number of leaves
- $w_j$ = leaf weights

### Gradient Boosting
$$f_t(x) = f_{t-1}(x) + \eta g_t(x)$$

Where:
- $\eta$ = learning rate
- $g_t$ = new tree minimizing loss

## Key Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|----------|
| `max_depth` | 6 | Controls tree complexity |
| `learning_rate` | 0.1 | Shrinkage factor |
| `n_estimators` | 200 | Number of boosting rounds |
| `subsample` | 0.8 | Row subsampling ratio |
| `colsample_bytree` | 0.8 | Column subsampling ratio |
| `reg_alpha` | 0.5 | L1 regularization |
| `reg_lambda` | 1.0 | L2 regularization |

## Implementation
See `src/xgboost_model.py` for complete implementation.

## Advantages
- Fast training
- Handles missing values
- Feature importance
- Regularization built-in
- Parallel processing

## Training Process
1. Initialize predictions
2. For each iteration: calculate residuals, fit new tree, update predictions
3. Apply learning rate shrinkage
4. Early stopping based on validation performance
