import pandas as pd
from sklearn.model_selection import train_test_split

# from src.config import config

def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw Titanic train and test CSVs."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def split_features_target(
    cfg,
    df: pd.DataFrame, 
    target_col: str = "Survived",
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from target."""
    # columns_to_drop = config["preprocessing"]["columns_to_drop"]
    columns_to_drop = cfg.preprocessing.columns_to_drop
    X = df.drop(columns=columns_to_drop)
    y = df[target_col]
    # return train_test_split(X, y, test_size=config["training"]["test_size"], random_state=config["random_state"], stratify=y)
    return train_test_split(
        X, y,
        test_size=cfg.training.test_size,
        random_state=cfg.random_state,
        stratify=y
    )
