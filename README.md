# Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to churn using Logistic Regression and Random Forest.

## Project Overview

This project uses customer data to predict customer churn. It includes data generation, preprocessing, model training, prediction, and model evaluation.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Logistic Regression
- Random Forest

## Machine Learning Workflow

Customer Data → Preprocessing → Feature Engineering → Model Training → Prediction → Evaluation

## Models Used

### Logistic Regression
Used for binary classification to predict whether a customer will churn.

### Random Forest
Used for classification and feature importance analysis.

## Project Structure

```text
customer_churn_prediction/
├── src/
│   ├── generate_dataset.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
└── outputs/
    ├── classification_report_random_forest.txt
    └── confusion_matrix_random_forest.png
    Evaluation

The models are evaluated using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
How to Run
pip install pandas numpy scikit-learn matplotlib

Run the project scripts according to the project workflow.

Future Improvements
Improve model performance
Add more machine learning algorithms
Build a Streamlit web application
Deploy the model online
Use a larger real-world dataset
Author

Rajkumar

B.Tech Computer Science Engineering
IILM University, Greater Noida

License

MIT License

### 2. Commit section mein

**Commit message:**
```text
Add README for customer churn prediction project
