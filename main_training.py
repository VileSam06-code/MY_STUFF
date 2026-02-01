"""
Main Training Pipeline for Fraud Detection

Demonstrates the complete end-to-end fraud detection workflow using:
- Data loading and preprocessing
- Feature engineering for transaction data
- Training multiple models (XGBoost, Deep Learning, Ensemble)
- Model evaluation and comparison
- Results analysis

"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
SRC_PATH = Path(__file__).parent / 'src'
sys.path.insert(0, str(SRC_PATH))

def main():
    """
    Main execution function demonstrating the complete fraud detection pipeline.
    """
    
    try:
        # Step 1: Load and preprocess data
        
        # Load UPI fraud dataset
        data_path = Path(__file__).parent / 'data' / 'upi_fraud_dataset.csv'
        if data_path.exists():
            df = pd.read_csv(data_path)
            print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

            # Detect target column (some datasets use 'is_fraud' instead of 'fraud')
            possible_targets = ["fraud_risk", "fraud", "is_fraud", "label", "target"]
            target_col = None
            for col in possible_targets:
                if col in df.columns:
                    target_col = col
                    break

            if target_col is None:
                raise KeyError(
                    f"No fraud label column found. Expected one of {possible_targets}. "
                    f"Available columns: {list(df.columns)}"
                )

        else:
            logger.warning(f"Dataset not found at {data_path}")
            logger.info("Using synthetic data for demonstration...")
            # Create dummy data structure for demonstration
            df = pd.DataFrame()
        
        # Step 2: Feature Engineering
        
        # Step 3: REAL Model Training + Evaluation (Aligned with instructions.md)

        if df.empty:
            raise ValueError("Dataset not loaded. Put upi_fraud_dataset.csv inside data/")

        # ----------------------------
        # Separate Features + Target
        # ----------------------------
        target = target_col
        X = df.drop(columns=[target])
        y = df[target]

        # Drop leakage-prone identifiers
        X = X.drop(columns=["Id", "upi_number"], errors="ignore")

        # ----------------------------
        # Minimal Feature Engineering
        # ----------------------------
        if "trans_amount" in X.columns:
            X["log_amount"] = np.log1p(X["trans_amount"])

        # Risk encoding for category/state (allowed by dataset)
        if "category" in df.columns:
            cat_risk = df.groupby("category")[target].mean()
            X["category_risk"] = df["category"].map(cat_risk)

        if "state" in df.columns:
            state_risk = df.groupby("state")[target].mean()
            X["state_risk"] = df["state"].map(state_risk)

        # One-hot encode remaining categorical features
        X = pd.get_dummies(X, drop_first=True)

        # ----------------------------
        # Train/Test Split (Instruction requirement)
        # ----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        # ----------------------------
        # Model: Logistic Regression Baseline
        # ----------------------------
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        from sklearn.model_selection import cross_val_score

        print("\nTraining Logistic Regression Baseline...")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = LogisticRegression(class_weight="balanced", max_iter=1000)
        lr.fit(X_train_scaled, y_train)

        y_pred_lr = lr.predict(X_test_scaled)
        y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

        from sklearn.metrics import confusion_matrix

        # Confusion Matrix (Logistic Regression)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lr).ravel()
        print("\nConfusion Matrix (Logistic Regression)")
        print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        print()

        acc_lr = accuracy_score(y_test, y_pred_lr)
        f1_lr = f1_score(y_test, y_pred_lr)
        auc_lr = roc_auc_score(y_test, y_prob_lr)

        print(f"LogReg Accuracy: {acc_lr:.4f}")
        print(f"LogReg F1 Score: {f1_lr:.4f}")
        print(f"LogReg ROC-AUC: {auc_lr:.4f}")

        cv_lr = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring="f1")
        print(f"LogReg Mean CV F1: {cv_lr.mean():.4f}")

        # ----------------------------
        # Model: XGBoost Fraud Detector (Final)
        # ----------------------------
        from xgboost import XGBClassifier

        print("\nTraining XGBoost Fraud Model...")

        imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=imbalance_ratio,
            eval_metric="aucpr",
            random_state=42
        )

        xgb.fit(X_train, y_train)

        y_pred_xgb = xgb.predict(X_test)
        y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

        # Confusion Matrix (XGBoost)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_xgb).ravel()
        print("\nConfusion Matrix (XGBoost)")
        print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        f1_xgb = f1_score(y_test, y_pred_xgb)
        auc_xgb = roc_auc_score(y_test, y_prob_xgb)

        print(f"XGBoost Accuracy: {acc_xgb:.4f}")
        print(f"XGBoost F1 Score: {f1_xgb:.4f}")
        print(f"XGBoost ROC-AUC: {auc_xgb:.4f}")

        cv_xgb = cross_val_score(xgb, X_train, y_train, cv=5, scoring="f1")
        print(f"XGBoost Mean CV F1: {cv_xgb.mean():.4f}")

        # ----------------------------
        # Final Summary for Submission
        # ----------------------------
        print("\nFinal metrics")
        print("-" * 40)
        print(f"XGBoost Accuracy: {acc_xgb:.4f}")
        print(f"XGBoost F1 Score: {f1_xgb:.4f}")
        print(f"XGBoost ROC-AUC: {auc_xgb:.4f}")
        print(f"XGBoost Mean CV F1: {cv_xgb.mean():.4f}")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
