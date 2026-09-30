"""
preprocess.py
-------------
Shared preprocessing for the churn prediction pipeline.

- Binary Yes/No (and Male/Female) columns -> 0/1
- Multi-category columns (Contract, InternetService, PaymentMethod, ...)
  -> one-hot encoded
- Numeric columns (tenure, MonthlyCharges, TotalCharges) -> passed through
  (scaling is handled inside the model pipeline via StandardScaler for the
  logistic regression model; tree-based Random Forest does not need it)

build_preprocessor() returns a scikit-learn ColumnTransformer that can be
placed in a Pipeline ahead of any classifier, so train.py and predict.py
share the exact same transformation.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

BINARY_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
]

CATEGORICAL_FEATURES = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

# SeniorCitizen is already 0/1 in the source data -> treat as numeric/passthrough
PASSTHROUGH_FEATURES = ["SeniorCitizen"]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES + PASSTHROUGH_FEATURES


def build_preprocessor():
    """Returns a ColumnTransformer: scales numeric features and one-hot
    encodes every categorical/binary feature (binary columns are just
    2-category one-hot encodings, handled the same way as the rest)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("pass", "passthrough", PASSTHROUGH_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), BINARY_FEATURES + CATEGORICAL_FEATURES),
        ]
    )


def prepare_xy(df):
    """Split a raw dataframe into X (features) and y (binary churn label)."""
    X = df[ALL_FEATURES].copy()
    y = (df[TARGET_COLUMN] == "Yes").astype(int)
    return X, y
