"""
generate_dataset.py
--------------------
Builds a synthetic, offline telecom customer churn dataset in the same
shape as the well-known Telco Customer Churn dataset, so this project
runs fully offline and reproducibly (no Kaggle/internet download needed).

Rather than assigning churn randomly, each customer's churn probability
is computed from a realistic rule-of-thumb model of churn risk factors
(short tenure, month-to-month contracts, fiber internet with no tech
support, electronic check payment, high monthly charges, etc.), then a
Bernoulli draw + noise decides the actual label. This keeps the dataset
learnable (there IS real signal) but not trivially separable (~75-85%
accuracy is the realistic ceiling for classical models on this kind of
data, matching published benchmarks on the real Telco churn dataset).

Run:
    python src/generate_dataset.py
Produces:
    data/telecom_churn.csv
"""

import csv
import random
from pathlib import Path

random.seed(42)

N_CUSTOMERS = 2000
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "telecom_churn.csv"

CONTRACTS = ["Month-to-month", "One year", "Two year"]
CONTRACT_WEIGHTS = [0.55, 0.24, 0.21]
INTERNET = ["DSL", "Fiber optic", "No"]
INTERNET_WEIGHTS = [0.35, 0.45, 0.20]
PAYMENT = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
PAYMENT_WEIGHTS = [0.34, 0.19, 0.24, 0.23]
YES_NO = ["Yes", "No"]


def yes_no(p_yes=0.5):
    return "Yes" if random.random() < p_yes else "No"


def make_customer(customer_id):
    gender = random.choice(["Male", "Female"])
    senior = 1 if random.random() < 0.16 else 0
    partner = yes_no(0.48)
    dependents = yes_no(0.30 if partner == "Yes" else 0.10)

    tenure = int(random.betavariate(1.6, 2.2) * 72)  # skewed 0-72 months
    tenure = max(0, min(72, tenure))

    contract = random.choices(CONTRACTS, weights=CONTRACT_WEIGHTS)[0]
    internet = random.choices(INTERNET, weights=INTERNET_WEIGHTS)[0]

    phone_service = yes_no(0.90)
    multiple_lines = "No phone service" if phone_service == "No" else yes_no(0.42)

    if internet == "No":
        online_security = online_backup = device_protection = tech_support = "No internet service"
        streaming_tv = streaming_movies = "No internet service"
    else:
        online_security = yes_no(0.35)
        online_backup = yes_no(0.38)
        device_protection = yes_no(0.38)
        tech_support = yes_no(0.35)
        streaming_tv = yes_no(0.42)
        streaming_movies = yes_no(0.42)

    paperless_billing = yes_no(0.59)
    payment_method = random.choices(PAYMENT, weights=PAYMENT_WEIGHTS)[0]

    # Base monthly charge components
    base = 20.0
    if internet == "DSL":
        base += 20 + random.uniform(-3, 3)
    elif internet == "Fiber optic":
        base += 45 + random.uniform(-5, 5)
    if phone_service == "Yes":
        base += 5
    for svc in [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]:
        if svc == "Yes":
            base += random.uniform(4, 8)
    monthly_charges = round(max(18.0, base), 2)
    total_charges = round(monthly_charges * tenure * random.uniform(0.95, 1.05) + random.uniform(0, 20), 2)

    # ---- Churn probability model (the "ground truth" business logic) ----
    risk = 0.08
    if contract == "Month-to-month":
        risk += 0.34
    elif contract == "Two year":
        risk -= 0.24
    if tenure < 6:
        risk += 0.28
    elif tenure < 12:
        risk += 0.14
    elif tenure > 48:
        risk -= 0.20
    if internet == "Fiber optic":
        risk += 0.13
    if tech_support in ("No",):
        risk += 0.09
    if online_security in ("No",):
        risk += 0.06
    if payment_method == "Electronic check":
        risk += 0.11
    if monthly_charges > 85:
        risk += 0.07
    if senior == 1:
        risk += 0.05
    if partner == "Yes":
        risk -= 0.04
    if dependents == "Yes":
        risk -= 0.05
    if paperless_billing == "Yes":
        risk += 0.02

    risk += random.uniform(-0.025, 0.025)  # unobserved factors / noise
    risk = min(max(risk, 0.02), 0.95)
    churn = "Yes" if random.random() < risk else "No"

    return {
        "customerID": f"CUST-{customer_id:05d}",
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn,
    }


def main():
    rows = [make_customer(i + 1) for i in range(N_CUSTOMERS)]
    fieldnames = list(rows[0].keys())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_churn = sum(1 for r in rows if r["Churn"] == "Yes")
    print(f"Wrote {len(rows)} customers to {OUTPUT_PATH}")
    print(f"Churn rate: {n_churn}/{len(rows)} = {n_churn / len(rows):.1%}")


if __name__ == "__main__":
    main()
