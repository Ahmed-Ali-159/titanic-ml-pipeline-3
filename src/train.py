import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessor import build_preprocessor


def build_models() -> dict:
    preprocessor = build_preprocessor()
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=100, random_state=42),
                ),
            ]
        ),
    }


def train_and_save(models: dict, X_train, y_train, output_dir: str = "models/") -> dict:
    trained = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        path = f"{output_dir}{name}.pkl"
        joblib.dump(pipeline, path)
        print(f"[✓] Saved {name} → {path}")
        trained[name] = pipeline
    return trained
