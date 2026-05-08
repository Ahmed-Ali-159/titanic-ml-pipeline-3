import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessor import build_preprocessor

# import yaml

# with open("configs/config.yaml", "r") as f:
#     config = yaml.safe_load(f)

from src.config import config

def build_models() -> dict:
    preprocessor = build_preprocessor()
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=config["models"]["logistic_regression"]["max_iter"], random_state=config["random_state"])),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(n_estimators=config["models"]["random_forest"]["n_estimators"], random_state=config["random_state"]))
            ]
        )
    }


def train_and_save(models: dict, X_train, y_train, output_dir: str = config["training"]["output_dir"]) -> dict:
    trained = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        path = f"{output_dir}{name}.pkl"
        joblib.dump(pipeline, path)
        print(f"[✓] Saved {name} → {path}")
        trained[name] = pipeline
    return trained
