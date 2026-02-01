# Fraud Detection Dataset Implementation

## Overview

This module is my **hands-on implementation** of a fraud detection pipeline using a real UPI transaction dataset.

Through this project, I learned the complete machine learning workflow:

- Loading and preprocessing transaction data  
- Feature engineering for fraud patterns  
- Training baseline and advanced models  
- Evaluating results using proper fraud-relevant metrics  
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

Fraud probability is modeled as:

\[
P(y=1|x)=\sigma(w^Tx+b)
\]

This provides an interpretable baseline model.

---

### XGBoost (Final Model)

Boosting builds an ensemble of trees:

\[
f(x)=\sum_{t=1}^{T}\eta g_t(x)
\]

Class imbalance is handled using:

\[
scale\_pos\_weight=\frac{\#negative}{\#positive}
\]

---

## Evaluation Metrics

Fraud detection requires more than accuracy. I used:

### Confusion Matrix

\[
\begin{array}{c|cc}
 & Predicted Fraud & Predicted Normal \\
\hline
Actual Fraud & TP & FN \\
Actual Normal & FP & TN
\end{array}
\]

### Key Scores

Accuracy:

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

F1 Score:

\[
F1=\frac{2TP}{2TP+FP+FN}
\]

ROC Terms:

\[
TPR=\frac{TP}{TP+FN},\quad FPR=\frac{FP}{FP+TN}
\]

ROC-AUC measures ranking quality across thresholds.

---

## Running the Project

Run the full training and evaluation pipeline:

```bash
python3 main_training.py
