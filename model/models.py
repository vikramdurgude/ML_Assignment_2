import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}

def load_models():
    return {name: joblib.load(MODEL_DIR / filename)
            for name, filename in MODEL_FILES.items()}
