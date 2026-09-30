"""
predict.py
----------
Loads the saved churn model and predicts whether a customer will churn.

Usage:
    # Interactive mode: answer a few prompts about one customer
    python src/predict.py

    # Or edit the SAMPLE_CUSTOMER dict below and run non-interactively:
    python src/predict.py --sample
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from preprocess import ALL_FEATURES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

SAMPLE_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.50, "TotalCharges": 286.50,
}


def load_model():
    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Run `python src/train.py` first.")
    return joblib.load(model_path)


def predict_one(customer: dict, pipeline) -> tuple:
    row = pd.DataFrame([customer])[ALL_FEATURES]
    pred = pipeline.predict(row)[0]
    proba = pipeline.predict_proba(row)[0, 1]
    return ("Churn" if pred == 1 else "No Churn"), proba


def ask(prompt, options=None, cast=str, default=None):
    suffix = f" [{'/'.join(options)}]" if options else ""
    if default is not None:
        suffix += f" (default: {default})"
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        if options and raw not in options:
            print(f"  Please enter one of: {', '.join(options)}")
            continue
        try:
            return cast(raw)
        except ValueError:
            print("  Please enter a valid value.")


def interactive_customer():
    print("Enter the customer's details (press Enter to accept a default where shown):\n")
    c = {}
    c["gender"] = ask("Gender", ["Male", "Female"])
    c["SeniorCitizen"] = ask("Senior citizen? (1=yes, 0=no)", ["0", "1"], int)
    c["Partner"] = ask("Has partner?", ["Yes", "No"])
    c["Dependents"] = ask("Has dependents?", ["Yes", "No"])
    c["tenure"] = ask("Tenure (months with company)", cast=int)
    c["PhoneService"] = ask("Has phone service?", ["Yes", "No"])
    c["MultipleLines"] = ask("Multiple lines?", ["Yes", "No", "No phone service"],
                              default="No phone service" if c["PhoneService"] == "No" else "No")
    c["InternetService"] = ask("Internet service", ["DSL", "Fiber optic", "No"])
    no_internet = c["InternetService"] == "No"
    for field, label in [
        ("OnlineSecurity", "Online security?"), ("OnlineBackup", "Online backup?"),
        ("DeviceProtection", "Device protection?"), ("TechSupport", "Tech support?"),
        ("StreamingTV", "Streaming TV?"), ("StreamingMovies", "Streaming movies?"),
    ]:
        c[field] = "No internet service" if no_internet else ask(label, ["Yes", "No"])
    c["Contract"] = ask("Contract type", ["Month-to-month", "One year", "Two year"])
    c["PaperlessBilling"] = ask("Paperless billing?", ["Yes", "No"])
    c["PaymentMethod"] = ask("Payment method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    c["MonthlyCharges"] = ask("Monthly charges ($)", cast=float)
    c["TotalCharges"] = ask("Total charges to date ($)", cast=float)
    return c


def main():
    pipeline = load_model()

    if "--sample" in sys.argv:
        customer = SAMPLE_CUSTOMER
        print("Using built-in sample customer:\n", customer, "\n")
    else:
        customer = interactive_customer()

    label, proba = predict_one(customer, pipeline)
    print(f"\nPrediction: {label}")
    print(f"Churn probability: {proba:.1%}")


if __name__ == "__main__":
    main()
