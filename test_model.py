
from pathlib import Path

import joblib
import pandas as pd


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "outputs" / "customer_churn_test_data.csv"
MODEL_PATH = ROOT / "models" / "tuned_random_forest.joblib"
OUTPUT_PATH = ROOT / "outputs" / "customer_churn_test_predictions.csv"


# --------------------------------------------------
# 2. Expected columns from the cleaned dataset
# --------------------------------------------------
expected_columns = [
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "EstimatedSalary",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "Exited",
    "AgeGroup",
    "CreditScoreGroup",
    "BalanceGroup",
]


# --------------------------------------------------
# 3. Load and validate the test data
# --------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Test CSV not found: {DATA_PATH}\n"
        "Check that customer_churn_test_data.csv "
        "is inside the outputs folder."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("CUSTOMER CHURN - SAMPLE TEST DATA")
print("=" * 80)

print(f"\nDataset shape: {df.shape}")
print(f"Number of customers: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

if df.columns.tolist() != expected_columns:
    print("\nERROR: The columns do not match the cleaned dataset.")
    print("\nExpected columns:")
    print(expected_columns)
    print("\nColumns found:")
    print(df.columns.tolist())
    raise ValueError("Please correct the test CSV column names/order.")

print("\nAll 16 columns match the cleaned dataset.")


# --------------------------------------------------
# 4. Display all input records
# --------------------------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 30)

print("\n" + "=" * 80)
print("ALL CUSTOMER RECORDS - BEFORE PREDICTION")
print("=" * 80)

print(df.to_string(index=False))


# --------------------------------------------------
# 5. Load the saved Random Forest model
# --------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Saved model not found: {MODEL_PATH}"
    )

saved_model = joblib.load(MODEL_PATH)

model = saved_model["model"]
feature_columns = saved_model["feature_columns"]

print("\n" + "=" * 80)
print("SAVED MODEL LOADED SUCCESSFULLY")
print("=" * 80)

print("Model:", type(model).__name__)
print("Expected encoded features:", len(feature_columns))


# --------------------------------------------------
# 6. Select the original model input features
# --------------------------------------------------
# CustomerId, Surname, Exited and grouping columns
# are not used as model inputs.

input_columns = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "EstimatedSalary",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
]

X = df[input_columns].copy()


# --------------------------------------------------
# 7. Encode categorical features
# --------------------------------------------------
X = pd.get_dummies(
    X,
    columns=["Geography", "Gender"],
    drop_first=True,
    dtype=int,
)

# Match the exact feature columns used during training
X = X.reindex(columns=feature_columns, fill_value=0)

print("\nEncoded model input shape:", X.shape)


# --------------------------------------------------
# 8. Predict customer churn
# --------------------------------------------------
predictions = model.predict(X)

results = df.copy()

results["Predicted_Exited"] = predictions

results["Predicted_Status"] = results[
    "Predicted_Exited"
].map({
    0: "Stayed",
    1: "Churned",
})


# --------------------------------------------------
# 9. Display every customer and prediction
# --------------------------------------------------
print("\n" + "=" * 100)
print("COMPLETE CUSTOMER RECORDS WITH MODEL PREDICTIONS")
print("=" * 100)

print(results.to_string(index=False))


# --------------------------------------------------
# 10. Display a simple prediction summary
# --------------------------------------------------
print("\n" + "=" * 80)
print("PREDICTION SUMMARY")
print("=" * 80)

print("Total customers tested:", len(results))
print(
    "Predicted to stay:",
    (results["Predicted_Exited"] == 0).sum()
)
print(
    "Predicted to churn:",
    (results["Predicted_Exited"] == 1).sum()
)

print("\nCustomer-wise predictions:")

print(
    results[
        [
            "CustomerId",
            "Surname",
            "Age",
            "Geography",
            "NumOfProducts",
            "Predicted_Exited",
            "Predicted_Status",
        ]
    ].to_string(index=False)
)


# --------------------------------------------------
# 11. Save complete records and predictions
# --------------------------------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

results.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 80)
print("TESTING COMPLETED")
print("=" * 80)

print("Complete predictions saved to:")
print(OUTPUT_PATH)
