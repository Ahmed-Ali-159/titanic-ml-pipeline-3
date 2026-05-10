import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessor import build_preprocessor

# import yaml

# with open("configs/config.yaml", "r") as f:
#     config = yaml.safe_load(f)

from src.config import config

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}

def build_model() -> dict:
    preprocessor = build_preprocessor()

    model_name = config["model"]["active_model"]

    model_class = MODEL_REGISTRY[model_name]

    model_params = config["model"][model_name]

    # Equivalent to: LogisticRegression(max_iter=1000, random_state=42) or RandomForestClassifier(n_estimators=100, random_state=42)
    classifier = model_class(
        **model_params,
        random_state=config["random_state"]
    )

    # return {
    #     "logistic_regression": Pipeline(
    #         [
    #             ("preprocessor", preprocessor),
    #             ("classifier", LogisticRegression(max_iter=config["models"]["logistic_regression"]["max_iter"], random_state=config["random_state"])),
    #         ]
    #     ),
    #     "random_forest": Pipeline(
    #         [
    #             ("preprocessor", preprocessor),
    #             ("classifier", RandomForestClassifier(n_estimators=config["models"]["random_forest"]["n_estimators"], random_state=config["random_state"]))
    #         ]
    #     )
    # }

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return model_name, pipeline


def train_and_save(model_name: str, pipeline: Pipeline, X_train, y_train):
    
    # Ensure output directory exists
    output_dir = config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Train each model and save to disk
    # trained = {}
    # for name, pipeline in models.items():
    #     pipeline.fit(X_train, y_train)
    #     path = f"{output_dir}{name}.pkl"
    #     joblib.dump(pipeline, path)
    #     print(f"[✓] Saved {name} → {path}")
    #     trained[name] = pipeline
    # return trained

    # Train the model
    pipeline.fit(X_train, y_train)
    path = f"{output_dir}{model_name}.pkl"
    joblib.dump(pipeline, path)
    print(f"[✓] Saved {model_name} → {path}")
    return pipeline
