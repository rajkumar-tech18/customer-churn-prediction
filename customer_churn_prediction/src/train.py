"""
train.py
--------
Trains and compares two classifiers for telecom customer churn prediction:

    1. Logistic Regression (interpretable baseline; coefficients show
       which factors push churn risk up or down)
    2. Random Forest (typically higher accuracy; also yields a feature
       importance ranking)

Steps:
    1. Load data/telecom_churn.csv
    2. Train/test split (80/20, stratified on churn)
    3. Preprocess (scale numeric, one-hot encode categorical) via a
       scikit-learn Pipeline so the same transform is reused at predict time
    4. Train both models, evaluate accuracy / precision / recall / F1 / ROC-AUC
    5. Save a confusion matrix, ROC curve, and feature-importance plot
    6. Save the best-performing model (by ROC-AUC) with joblib

Run:
    python src/train.py
"""

import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parent))
from preprocess import build_preprocessor, prepare_xy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telecom_churn.csv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python src/generate_dataset.py` first."
        )
    return pd.read_csv(DATA_PATH)


def evaluate_model(name, pipeline, X_test, y_test):
    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds, target_names=["No Churn", "Churn"], zero_division=0)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}   ROC-AUC: {auc:.4f}")
    print(report)

    slug = name.replace(" ", "_").lower()
    (OUTPUTS_DIR / f"classification_report_{slug}.txt").write_text(
        f"{name}\nAccuracy: {acc:.4f}\nROC-AUC: {auc:.4f}\n\n{report}"
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / f"confusion_matrix_{slug}.png", dpi=150)
    plt.close(fig)

    return acc, auc, proba


def plot_roc(y_test, results):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, proba in results.items():
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "roc_curve_comparison.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(rf_pipeline, top_n=15):
    preprocessor = rf_pipeline.named_steps["preprocess"]
    model = rf_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    order = np.argsort(importances)[::-1][:top_n]
    top_features = feature_names[order]
    top_importances = importances[order]

    # Clean up sklearn's prefixed feature names for readability
    clean_names = [f.split("__")[-1] for f in top_features]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(clean_names[::-1], top_importances[::-1], color="#1E2761")
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Churn Predictors (Random Forest)")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} customers. Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")

    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = {}
    roc_data = {}

    print("\nTraining Logistic Regression...")
    t0 = time.time()
    logreg_pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    logreg_pipeline.fit(X_train, y_train)
    print(f"  done in {time.time() - t0:.2f}s")
    acc, auc, proba = evaluate_model("Logistic Regression", logreg_pipeline, X_test, y_test)
    results["Logistic Regression"] = (logreg_pipeline, acc, auc)
    roc_data["Logistic Regression"] = proba

    print("\nTraining Random Forest...")
    t0 = time.time()
    rf_pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    rf_pipeline.fit(X_train, y_train)
    print(f"  done in {time.time() - t0:.2f}s")
    acc, auc, proba = evaluate_model("Random Forest", rf_pipeline, X_test, y_test)
    results["Random Forest"] = (rf_pipeline, acc, auc)
    roc_data["Random Forest"] = proba

    plot_roc(y_test, roc_data)
    plot_feature_importance(rf_pipeline)

    # Best model by ROC-AUC (more robust than accuracy for imbalanced churn data)
    best_name, (best_pipeline, best_acc, best_auc) = max(results.items(), key=lambda kv: kv[1][2])
    print(f"\nBest model: {best_name} (accuracy={best_acc:.4f}, ROC-AUC={best_auc:.4f})")

    joblib.dump(best_pipeline, MODELS_DIR / "best_model.joblib")
    (MODELS_DIR / "best_model_name.txt").write_text(best_name)

    summary_lines = ["Model comparison:"]
    for name, (_, acc, auc) in results.items():
        summary_lines.append(f"  {name}: accuracy={acc:.4f}, ROC-AUC={auc:.4f}")
    summary_lines.append(f"\nSelected best model: {best_name}")
    (OUTPUTS_DIR / "model_comparison.txt").write_text("\n".join(summary_lines))

    print(f"\nSaved model -> {MODELS_DIR / 'best_model.joblib'}")
    print(f"Saved reports/plots -> {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
