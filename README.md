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

- Accuracy = (TP + TN) / (TP + TN + FP + FN)  
- Precision = TP / (TP + FP)  
- Recall = TP / (TP + FN)  
- F1 Score = 2TP / (2TP + FP + FN)  

ROC-AUC measures ranking quality across thresholds.

---

## Results (Real Dataset Output)

Dataset used:

- Samples: **2666**
- Features: **12**

---

### Logistic Regression Baseline

Confusion Matrix:

- TN = 202  
- FP = 16  
- FN = 83  
- TP = 233  

Metrics:

- Accuracy: **0.8146**  
- F1 Score: **0.8248**  
- ROC-AUC: **0.8882**  
- Mean CV F1 (5-fold): **0.8296**  

---

### XGBoost Fraud Model (Final)

Confusion Matrix:

- TN = 210  
- FP = 8  
- FN = 10  
- TP = 306  

Metrics:

- Accuracy: **0.9663**  
- F1 Score: **0.9714**  
- ROC-AUC: **0.9968**  
- Mean CV F1 (5-fold): **0.9656**  

---

Final Selected Model: **XGBoost**

---

## Running the Project

Run the full training and evaluation pipeline:

```bash
python3 main_training.py
