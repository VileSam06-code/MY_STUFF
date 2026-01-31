# Ensemble Methods for Fraud Detection

## Overview
Ensemble methods combine multiple models to improve prediction accuracy and robustness.

## Voting Classifier

### Simple Voting
$$\hat{y} = \text{mode}(f_1(x), f_2(x), ..., f_k(x))$$

### Weighted Voting  
$$\hat{y} = \arg\max_c \sum_i w_i \cdot \mathbb{1}(f_i(x) = c)$$

Where $w_i$ are learnable or fixed weights.

## Averaging
$$\hat{y} = \frac{1}{k}\sum_i p_i(x)$$

For probability predictions: average then apply threshold.

## Stacking

Meta-learner trained on base model predictions:
1. Base models: XGBoost, Neural Network
2. Meta-features: Predictions from base models
3. Meta-learner: Logistic regression on meta-features

## Implementation
See `src/ensemble_model.py` for complete code.

## Advantages
- Combines strengths of multiple models
- Reduces variance and bias
- Improves generalization
- More robust predictions

## Model Combination Strategy
- XGBoost: Fast, interpretable
- Neural Network: Captures complex patterns  
- Ensemble: Best of both
