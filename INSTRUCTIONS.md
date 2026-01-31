# Fraud Detection Pipeline - Complete Instructions for Students

This document provides step-by-step instructions for using the fraud detection package to train, evaluate, and deploy fraud detection models.

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Dataset Setup](#dataset-setup)
4. [Quick Start](#quick-start)
5. [Running Examples](#running-examples)
6. [Understanding the Models](#understanding-the-models)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages:
- **Data Processing**: pandas, numpy, scipy
- **ML Models**: scikit-learn, xgboost, tensorflow, keras
- **Sampling**: imbalanced-learn (SMOTE)
- **Visualization**: matplotlib, seaborn
- **Jupyter**: jupyter, ipython

### Step 2: Verify Installation

```bash
python -c "import xgboost; import tensorflow; print('All libraries installed successfully!')"
```

---

## Project Structure

```
07-dataset-implementation/
├── src/                          # Main source code
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration file (EDIT THIS!)
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── feature_engineering.py   # Feature engineering
│   ├── xgboost_model.py         # XGBoost implementation
│   ├── deep_learning_model.py   # Deep Learning implementation
│   ├── ensemble_model.py        # Ensemble methods
│   ├── model_evaluation.py      # Evaluation metrics
│   ├── visualization.py         # Visualization tools
│   └── deployment_pipeline.py   # Production pipeline
│
├── data/                         # Dataset folder
│   └── upi_fraud_dataset.csv    # UPI fraud dataset (place here)
│
├── models/                       # Trained models (auto-created)
├── logs/                         # Log files (auto-created)
├── config/                       # Configuration files
├── docs/                         # Documentation
│
├── main_training.py             # Complete training pipeline
├── example_usage.py             # Step-by-step examples
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview
└── INSTRUCTIONS.md              # This file
```

---

## Dataset Setup

### Obtaining the Dataset

1. **Place the UPI fraud dataset** in `data/upi_fraud_dataset.csv`

### Expected Dataset Format

The CSV file should contain the following columns:

```
trans_hour (int)      - Hour of transaction (0-23)
trans_day (int)       - Day of month (1-31)
trans_month (int)     - Month (1-12)
trans_year (int)      - Year
category (str)        - Merchant category code
upi_number (str)      - UPI transaction ID (optional, dropped during preprocessing)
age (int)             - Customer age
trans_amount (float)  - Transaction amount
state (str)           - State code
zip (str)             - ZIP code
fraud_risk (int)      - Target variable (0=legitimate, 1=fraud)
```

### Data Statistics
- **Expected samples**: 2000-10000+
- **Fraud rate**: 5-15% (imbalanced)
- **Features**: ~10-15 numerical and categorical

---

## Quick Start

### Option 1: Run Step-by-Step Examples (RECOMMENDED FOR LEARNING)

```bash
python example_usage.py
```

This runs 4 examples:
1. Load and preprocess data
2. Train XGBoost model
3. Train Deep Learning model (optional)
4. Evaluate model performance

### Option 2: Run Complete Training Pipeline

```bash
python main_training.py
```

This demonstrates the complete workflow end-to-end with all 7 steps.

---

## Running Examples

### Example 1: Basic Data Loading

```python
from src.data_loader import DataLoader
from src.config import DATA_CONFIG

# Load data
loader = DataLoader(DATA_CONFIG['filepath'])
df = loader.load_data()

# Preprocess
X_train, X_test, y_train, y_test = loader.preprocess(
    test_size=0.2,
    apply_smote=True
)

print(f"Training set shape: {X_train.shape}")
print(f"Fraud rate: {y_train.mean():.2%}")
```

### Example 2: Training XGBoost

```python
from src.xgboost_model import XGBoostFraudDetector
from src.config import XGBOOST_CONFIG

# Initialize
model = XGBoostFraudDetector(
    learning_rate=XGBOOST_CONFIG['learning_rate'],
    max_depth=XGBOOST_CONFIG['max_depth']
)

# Train
model.train(X_train, y_train, X_test, y_test)

# Predict
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

### Example 3: Training Deep Learning Model

```python
from src.deep_learning_model import DeepLearningDetector
from src.config import DEEP_LEARNING_CONFIG

# Initialize
model = DeepLearningDetector(
    input_dim=X_train.shape[1],
    hidden_units=DEEP_LEARNING_CONFIG['hidden_units']
)

# Build and train
model.build_model()
model.train(X_train, y_train, X_test, y_test, epochs=50)

# Evaluate
y_pred_proba = model.predict_proba(X_test)
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

### Example 4: Ensemble Models

```python
from src.ensemble_model import EnsembleDetector
from src.config import ENSEMBLE_CONFIG

# Initialize with base models
ensemble = EnsembleDetector(
    method='stacking',
    base_models=['xgboost', 'deep_learning']
)

# Train
ensemble.train(X_train, y_train, X_test, y_test)

# Predict
y_pred = ensemble.predict(X_test)
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.4f}")
```

---

## Understanding the Models

### 1. XGBoost (Gradient Boosting)

**When to use**: Fast training, interpretable results, handles missing data

**Key parameters**:
- `learning_rate`: Controls step size (default: 0.1)
- `max_depth`: Tree depth (default: 6)
- `n_estimators`: Number of trees (default: 200)

**Advantages**:
- Very fast training
- Feature importance ranking
- Handles imbalanced data well
- Good baseline model

### 2. Deep Learning (Neural Networks)

**When to use**: Complex patterns, non-linear relationships

**Architecture**:
- Input layer: # features
- Hidden layers: [128, 64, 32] units with ReLU
- Dropout: 0.3 (regularization)
- Output: Sigmoid (binary classification)

**Advantages**:
- Captures non-linear patterns
- Flexible architecture
- Can handle large datasets

### 3. Ensemble Methods

**Stacking approach**:
1. Train base models (XGBoost + Deep Learning)
2. Use predictions as features for meta-learner
3. Meta-learner combines predictions

**Advantages**:
- Combines strengths of multiple models
- Usually better performance
- Reduces overfitting

---

## Configuration

### Edit `src/config.py` to customize:

**Data settings**:
```python
DATA_CONFIG = {
    'filepath': 'path/to/upi_fraud_dataset.csv',
    'test_size': 0.2,
    'apply_smote': True,
}
```

**XGBoost hyperparameters**:
```python
XGBOOST_CONFIG = {
    'learning_rate': 0.1,
    'max_depth': 6,
    'n_estimators': 200,
}
```

**Deep Learning settings**:
```python
DEEP_LEARNING_CONFIG = {
    'epochs': 50,
    'batch_size': 32,
    'hidden_units': [128, 64, 32],
}
```

---

## Troubleshooting

### Error: "No module named 'xgboost'"
**Solution**: Run `pip install xgboost`

### Error: "FileNotFoundError: dataset not found"
**Solution**: Ensure dataset is in `data/upi_fraud_dataset.csv`

### Error: "No module named 'tensorflow'"
**Solution**: Run `pip install tensorflow` (for Deep Learning only)

### Model training is slow
**Solution**: 
- Reduce `n_estimators` in config.py
- Use GPU: Set `gpu_id` in XGBOOST_CONFIG
- Reduce dataset size for testing

### Out of memory errors
**Solution**:
- Reduce `batch_size` in DEEP_LEARNING_CONFIG
- Use smaller dataset subset
- Disable SMOTE in DATA_CONFIG

---

## Key Learning Points

1. **Feature Engineering** is critical for fraud detection
2. **Class Imbalance** requires special handling (SMOTE, class weights)
3. **Ensemble Methods** often outperform individual models
4. **Threshold Tuning** is important for business metrics
5. **Model Evaluation** should use business metrics, not just accuracy

---

## Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Place dataset in `data/upi_fraud_dataset.csv`
3. ✅ Run examples: `python example_usage.py`
4. ✅ Review main_training.py
5. ✅ Customize config.py for your needs
6. ✅ Train full pipeline: `python main_training.py`

---

## Support

For issues or questions:
1. Check the README.md
2. Review src/config.py comments
3. Check function docstrings in source files
4. Review example_usage.py for working examples

---

**Last Updated**: January 2026
**Designed for**: Engineering Students (IIT & Others)
**Level**: Intermediate ML
