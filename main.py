from src.data_loader import load_data, split_features_target
from src.evaluate import evaluate_model
from src.train import build_model, train_and_save

import hydra
from omegaconf import DictConfig

import mlflow
import os
from dotenv import load_dotenv


@hydra.main(
    version_base=None,
    # config_path="../configs",
    config_path="configs",
    config_name="config"
)


def main(cfg: DictConfig) -> None:

    # Set up MLflow
    load_dotenv()       # loads .env file
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("titanic-ml-pipeline")
    
    # Loading data
    print("=== Loading Data ===")
    # train_df, _ = load_data(config["data"]["train_path"], config["data"]["test_path"])
    train_df, _ = load_data(
        cfg.data.train_path,
        cfg.data.test_path
    )

    # Split data into train, and validation sets
    # X_train, X_val, y_train, y_val = split_features_target(df=train_df, target_col=config["preprocessing"]["target_col"])
    X_train, X_val, y_train, y_val = split_features_target(cfg=cfg, df=train_df, target_col=cfg.preprocessing.target_col)

    # # Train models and save them to disk
    # print("\n=== Training Models ===")
    # models = build_models()
    # trained = train_and_save(models, X_train, y_train)

    # # Evaluate models on validation set
    # print("\n=== Evaluating Models ===")
    # evaluate_models(trained, X_val, y_val)

     # Train models and save them to disk
    print("\n=== Training Model ===")
    
    # model_name, pipeline = build_model()
    model_name, pipeline = build_model(cfg)
    # trained_pipeline = train_and_save(cfg=cfg, model_name=model_name, pipeline=pipeline, X_train=X_train, y_train=y_train)
    trained_pipeline, run_id = train_and_save(cfg=cfg, model_name=model_name, pipeline=pipeline, X_train=X_train, y_train=y_train)

    # Evaluate models on validation set
    print("\n=== Evaluating Model ===")
    # evaluate_model(model_name, trained_pipeline, X_val, y_val)
    evaluate_model(model_name, trained_pipeline, X_val, y_val, run_id=run_id)


if __name__ == "__main__":
    main()
