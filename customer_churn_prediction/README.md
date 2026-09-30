# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer will
**churn** (leave the company) based on their account details, services
subscribed, contract type, and billing information. Two classifiers —
**Logistic Regression** and **Random Forest** — are trained and compared.

## Project Structure

```
customer_churn_prediction/
├── README.md
├── requirements.txt
├── data/
│   └── telecom_churn.csv         # labeled customer dataset
├── models/                        # created by train.py
│   ├── best_model.joblib
│   └── best_model_name.txt
├── outputs/                       # created by train.py
│   ├── classification_report_logistic_regression.txt
│   ├── classification_report_random_forest.txt
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── roc_curve_comparison.png
│   ├── feature_importance.png
│   └── model_comparison.txt
└── src/
    ├── generate_dataset.py        # builds data/telecom_churn.csv
    ├── preprocess.py              # shared feature encoding/scaling
    ├── train.py                   # trains, evaluates, saves the best model
    └── predict.py                 # predicts churn for one customer
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+.

## How to Run

### 1. Generate the dataset

```bash
python src/generate_dataset.py
```

Writes `data/telecom_churn.csv` (2,000 customers). A copy is already
included, so this is only needed if you want to regenerate it.

### 2. Train and evaluate the models

```bash
python src/train.py
```

This will:
- Load the dataset and split it 80/20 (stratified on churn)
- Encode categorical features (one-hot) and scale numeric features inside
  a scikit-learn `Pipeline`
- Train **Logistic Regression** and **Random Forest**
- Print accuracy, precision, recall, F1-score, and ROC-AUC for both
- Save confusion matrices, an ROC curve comparison, and a feature
  importance chart to `outputs/`
- Save the better-performing model (by ROC-AUC) to `models/`

### 3. Predict a customer's churn risk

```bash
# Interactive mode — answer prompts about one customer
python src/predict.py

# Or use the built-in sample high-risk customer
python src/predict.py --sample
```

Example output:
```
Prediction: Churn
Churn probability: 82.9%
```

## Approach

### Why a synthetic dataset?

The well-known Telco Customer Churn dataset is normally downloaded from
Kaggle, which requires internet access. To keep this project fully
offline and reproducible, `generate_dataset.py` builds an equivalent
dataset locally, matching the same columns (demographics, services,
contract, billing) as the real dataset.

Each customer's true churn probability is computed from a **rule-based
risk model** grounded in well-documented, real-world churn drivers:

| Factor | Effect on churn risk |
|---|---|
| Month-to-month contract | + strong increase |
| Two-year contract | − strong decrease |
| Tenure < 6 months | + strong increase |
| Tenure > 48 months | − decrease |
| Fiber optic internet | + increase |
| No tech support / no online security | + increase |
| Electronic check payment | + increase |
| High monthly charges (>$85) | + slight increase |
| Senior citizen | + slight increase |
| Has partner / dependents | − slight decrease |

A small amount of random noise is added before the final Yes/No label is
drawn, so the task is realistically difficult rather than perfectly
separable — the same principle used in the accompanying Document
Classification project.

### Pipeline

1. **Preprocessing** (`preprocess.py`): numeric features (tenure, monthly
   charges, total charges) are standardized; all categorical and binary
   features are one-hot encoded — wrapped in a scikit-learn
   `ColumnTransformer` so training and prediction use identical logic.
2. **Modeling**: two classifiers are trained and compared —
   - **Logistic Regression** — interpretable, coefficients directly show
     which factors raise or lower churn odds.
   - **Random Forest** — a stronger non-linear model that also yields a
     feature-importance ranking.
3. **Evaluation**: accuracy, precision/recall/F1 per class, ROC-AUC, and
   confusion matrices on a held-out 20% test split.
4. **Model selection**: the model with the higher ROC-AUC (a better metric
   than raw accuracy here, since churn classes are imbalanced) is saved
   for prediction.

### Results (on the included dataset)

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 72.5% | 0.795 |
| **Random Forest (selected)** | **73.3%** | **0.796** |

Both models comfortably beat the 58% majority-class baseline (always
predicting "No Churn"). The feature importance chart confirms the model
learned the same drivers used to generate the data: **contract type and
tenure are by far the strongest predictors**, followed by billing amount
and internet/support services — mirroring real-world churn analysis.

## Possible Extensions

- Swap in the real Telco Customer Churn dataset (from Kaggle) when
  internet access is available — the CSV just needs the same column
  names.
- Add gradient boosting (XGBoost/LightGBM) as a third model.
- Perform hyperparameter tuning via grid/random search.
- Convert probability output into a **customer risk score dashboard**
  (e.g. a simple Streamlit app) for a business-facing demo.
- Add SHAP values for per-customer explanation of *why* the model
  predicted churn (useful for retention teams).
