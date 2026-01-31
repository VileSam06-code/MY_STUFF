# Advanced Feature Engineering for Fraud Detection

## Overview
Feature engineering is crucial for fraud detection as it transforms raw transaction data into meaningful signals.

## Feature Categories

### 1. Velocity Features
**Definition**: Measure transaction frequency and monetary activity within time windows.

**Examples**:
- Transactions per day/week/month
- Total amount per customer in time window
- Average transaction amount

**Mathematical Formula**:
$$v_t = \frac{\text{count of transactions in window}}{\text{window duration}}$$

### 2. Deviation Features
**Definition**: Detect unusual behavior using statistical anomalies.

**Z-Score Formula**:
$$z = \frac{x - \mu}{\sigma}$$

Where:
- $x$ = transaction value
- $\mu$ = mean
- $\sigma$ = standard deviation

### 3. Aggregation Features
**Definition**: Customer-level statistics aggregated over transactions.

- Max/Min/Mean transaction amount
- Standard deviation of amounts
- Number of unique merchants

### 4. Temporal Features
**Definition**: Time-based patterns in fraudulent behavior.

- Hour of day (0-23)
- Day of week (0-6)
- Is weekend flag
- Is night hours flag
- Month of year

### 5. Interaction Features
**Definition**: Combinations of features capturing non-linear relationships.

**Product Interaction**:
$$f_{interaction} = f_1 \times f_2$$

**Ratio Interaction**:
$$f_{ratio} = \frac{f_1}{f_2 + \epsilon}$$

## Implementation
See `src/feature_engineering.py` for complete implementation.

## Best Practices
1. Handle missing values before feature engineering
2. Scale features appropriately
3. Avoid data leakage from test data
4. Monitor feature importance
5. Document all engineered features
