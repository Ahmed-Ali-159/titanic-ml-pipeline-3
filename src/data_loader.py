# import pandas as pd
# from sklearn.model_selection import train_test_split


# def load_data(train_path: str) -> tuple:
#     df = pd.read_csv(train_path)
#     X = df.drop(columns=["Survived", "PassengerId"])
#     y = df["Survived"]
#     return train_test_split(X, y, test_size=0.2, random_state=42)

############################################

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw Titanic train and test CSVs."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def split_features_target(
    df: pd.DataFrame, target_col: str = "Survived"
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from target."""
    columns_to_drop = [target_col, "PassengerId", "Name", "Ticket", "Cabin"]
    X = df.drop(columns=columns_to_drop)
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
