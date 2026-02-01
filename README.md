# Fraud Detection Dataset Implementation

## Overview

This module is my hands-on implementation of a fraud detection pipeline using a real UPI transaction dataset.

Through this project, I learned the complete ML workflow:

- Loading and preprocessing transaction data  
- Feature engineering for fraud patterns  
- Training baseline and advanced models  
- Evaluating results using fraud-relevant metrics  
- Validating predictions using confusion matrices  

---

## Dataset

The dataset contains UPI transaction records with:

- Amount (`trans_amount`)  
- Time features (`hour`, `day`, `month`, `year`)  
- Merchant category (`category`)  
- State/location (`state`)  
- Target label: `fraud_risk`  

---

## Models Implemented

### Logistic Regression (Baseline)

Logistic Regression models fraud probability as:

P(y=1|x) = sigmoid(wᵀx + b)

This provides an interpretable baseline.

---

### XGBoost (Final Model)

XGBoost builds an ensemble of decision trees:

f(x) = Σ η · gₜ(x)

Class imbalance is handled with:

scale_pos_weight = (#negative / #positive)

---

## Evaluation Metrics

Fraud detection requires more than accuracy. I used:

### Confusion Matrix

|               | Predicted Fraud | Predicted Normal |
|--------------|----------------|-----------------|
| Actual Fraud  | TP             | FN              |
| Actual Normal | FP             | TN              |

### Key Scores

Accuracy:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision:

Precision = TP / (TP + FP)

Recall:

Recall = TP / (TP + FN)

F1 Score:

F1 = 2TP / (2TP + FP + FN)

ROC Terms:

TPR = TP / (TP + FN)  
FPR = FP / (FP + TN)

ROC-AUC measures ranking quality across thresholds.

---

## Running the Project

Run the full training and evaluation pipeline:

```bash
python3 main_training.py
