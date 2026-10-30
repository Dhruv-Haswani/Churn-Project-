
from pathlib import Path
import joblib

model_path = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "tuned_random_forest.joblib"
)

if model_path.exists():
    saved = joblib.load(model_path)

    print("Saved model loaded successfully!")
    print("Model type:", type(saved["model"]).__name__)
    print("Best parameters:", saved["best_parameters"])
    print("Number of features:", len(saved["feature_columns"]))
else:
    print("Model file not found:", model_path)
