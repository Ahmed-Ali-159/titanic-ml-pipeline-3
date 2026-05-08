from src.data_loader import load_data, split_features_target
from src.evaluate import evaluate_models
from src.train import build_models, train_and_save

from src.config import config


def main() -> None:

    # Loading data
    print("=== Loading Data ===")
    train_df, _ = load_data(config["data"]["train_path"], config["data"]["test_path"])

    # Split data into train, and validation sets
    X_train, X_val, y_train, y_val = split_features_target(df=train_df, target_col=config["preprocessing"]["target_column"])

    # Train models and save them to disk
    print("\n=== Training Models ===")
    models = build_models()
    trained = train_and_save(models, X_train, y_train)

    # Evaluate models on validation set
    print("\n=== Evaluating Models ===")
    evaluate_models(trained, X_val, y_val)


if __name__ == "__main__":
    main()
