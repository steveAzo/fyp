# Data Module

## Purpose
Dataset loading, preprocessing, and train-test splitting for drift detection experiments.

## Usage

```python
from data.loader import DatasetLoader

# Load, preprocess, and split
loader = DatasetLoader()
loader.load_credit_default().preprocess().train_test_split()

X_train, y_train = loader.get_training_data()
X_test, y_test = loader.get_test_data()
```

## Files

### `loader.py`
**Class**: `DatasetLoader`

**Methods**:
- `load_credit_default()` - Load UCI credit default dataset
- `preprocess()` - Standardize features using StandardScaler
- `train_test_split(test_size=0.2)` - 80-20 split
- `get_training_data()` - Return X_train, y_train
- `get_test_data()` - Return X_test, y_test

## Datasets (To Add Later)

1. **Credit Default** (30k samples) - Starting dataset
2. **Customer Churn** (7k samples) - Smaller, faster prototyping
3. **Predictive Maintenance** (10k samples) - IoT-relevant
4. **Bike Sharing** (17k samples) - Regression task

## Data Files

Place raw datasets in this folder:
- `credit_raw.csv` - Credit card default (from Kaggle/UCI)
- `churn_raw.csv` - Customer churn
- `maintenance_raw.csv` - Predictive maintenance
- `bike_raw.csv` - Bike sharing demand

## Preprocessing

Each dataset needs:
1. **Handle missing values** (drop or impute)
2. **Remove duplicates**
3. **Encode categorical features** (OneHotEncoder or LabelEncoder)
4. **Standardize numerical features** (StandardScaler)
5. **Handle class imbalance** (if applicable)

Current implementation handles steps 4-5. Customize for each dataset.

