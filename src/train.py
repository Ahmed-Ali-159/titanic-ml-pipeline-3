import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessor import build_preprocessor

# from src.config import config

from omegaconf import OmegaConf


MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}

def build_model(cfg):
    preprocessor = build_preprocessor(cfg)

    model_name = cfg.model.name

    model_class = MODEL_REGISTRY[model_name]

    # model_params = cfg.model[model_name]
    # model_params = {
    #     k: v
    #     for k, v in cfg.model.items()
    #     if k != "name"
    # }

    model_params = {k: v for k, v in OmegaConf.to_container(cfg.model).items() if k != "name"}

    # Equivalent to: LogisticRegression(max_iter=1000, random_state=42) or RandomForestClassifier(n_estimators=100, random_state=42)
    classifier = model_class(
        **model_params,
        random_state=cfg.random_state
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return model_name, pipeline


def train_and_save(cfg, model_name: str, pipeline: Pipeline, X_train, y_train):
    
    # Ensure output directory exists
    # output_dir = config["training"]["output_dir"]
    output_dir = cfg.training.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Train the model
    pipeline.fit(X_train, y_train)
    path = f"{output_dir}{model_name}.pkl"
    joblib.dump(pipeline, path)
    print(f"[✓] Saved {model_name} → {path}")
    return pipeline
