
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "customer_churn_cleaned.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

# 2. Select features and target
features = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "EstimatedSalary", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember"
]

X = df[features]
y = df["Exited"]

# 3. Encode categorical features
X = pd.get_dummies(
    X,
    columns=["Geography", "Gender"],
    drop_first=True,
    dtype=int
)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 5. Define Random Forest and tuning grid
rf = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

# 6. Tune using training data only
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=1
)

print("\nTuning Random Forest...")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest parameters:")
print(grid.best_params_)

# 7. Evaluate on unseen test data
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nTEST RESULTS")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Stayed (0)", "Churned (1)"],
    zero_division=0
))

# 8. Save confusion matrix chart
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stayed (0)", "Churned (1)"]
)

display.plot(cmap="Blues", values_format="d")
plt.title("Tuned Random Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ca2_random_forest_confusion_matrix.png", dpi=300)
plt.close()

# 9. Save feature-importance chart
importance = pd.Series(
    best_model.feature_importances_,
    index=X_train.columns,
    name="Importance"
).sort_values(ascending=False)

importance.to_csv(OUTPUT_DIR / "ca2_feature_importance.csv")

top_features = importance.head(10).sort_values()

plt.figure(figsize=(9, 6))
top_features.plot(kind="barh")
plt.title("Tuned Random Forest - Top 10 Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ca2_random_forest_feature_importance.png", dpi=300)
plt.close()

# 10. Save test predictions
predictions = pd.DataFrame({
    "Actual_Exited": y_test.to_numpy(),
    "Predicted_Exited": y_pred
})
predictions.to_csv(OUTPUT_DIR / "ca2_test_predictions.csv", index=False)

# 11. Save evaluation metrics
metrics = pd.DataFrame([{
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_score": f1
}])
metrics.to_csv(OUTPUT_DIR / "ca2_evaluation_metrics.csv", index=False)

# 12. Save trained model and required feature columns
artifact = {
    "model": best_model,
    "feature_columns": X_train.columns.tolist(),
    "best_parameters": grid.best_params_
}
joblib.dump(artifact, MODEL_DIR / "tuned_random_forest.joblib")

print("\nSaved files:")
print("- models/tuned_random_forest.joblib")
print("- outputs/ca2_random_forest_confusion_matrix.png")
print("- outputs/ca2_random_forest_feature_importance.png")
print("- outputs/ca2_feature_importance.csv")
print("- outputs/ca2_test_predictions.csv")
print("- outputs/ca2_evaluation_metrics.csv")

print("\nCA2 Random Forest implementation completed successfully.")
