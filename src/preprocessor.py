from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# import yaml

# with open("configs/config.yaml", "r") as f:
#     config = yaml.safe_load(f)

from src.config import config


def build_preprocessor() -> ColumnTransformer:
    numeric_features = config["preprocessing"]["numerical_features"]
    categorical_features = config["preprocessing"]["categorical_features"]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=config["preprocessing"]["numerical_pipeline"]["imputer_strategy"])),
            ("scaler", config["preprocessing"]["numerical_pipeline"]["scaler"]()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=config["preprocessing"]["categorical_pipeline"]["imputer_strategy"], fill_value="missing")),
            ("encoder", config["preprocessing"]["categorical_pipeline"]["encoder"])
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor
