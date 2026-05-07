from src.data_loader import load_data, split_features_target
from src.evaluate import evaluate_models
from src.train import build_models, train_and_save


def main() -> None:

    # Loading data
    print("=== Loading Data ===")
    train_df, _ = load_data("data/raw/train.csv", "data/raw/test.csv")

    # Split data into train, and validation sets
    X_train, X_val, y_train, y_val = split_features_target(
        df=train_df, target_col="Survived"
    )

    # Train models and save them to disk
    print("\n=== Training Models ===")
    models = build_models()
    trained = train_and_save(models, X_train, y_train)

    # Evaluate models on validation set
    print("\n=== Evaluating Models ===")
    evaluate_models(trained, X_val, y_val)


if __name__ == "__main__":
    main()

    # print("\n=== Training Models ===")
    # models = build_models()
    # trained = train_and_save(models, X_train, y_train)

    # print("\n=== Evaluating Models ===")
    # evaluate_models(trained, X_test, y_test)
