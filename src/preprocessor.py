from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import config

SCALERS = {
    "standard": StandardScaler,
}

ENCODERS = {
    "onehot": OneHotEncoder,
}


def build_preprocessor() -> ColumnTransformer:
    numeric_features = config["preprocessing"]["numerical_features"]
    categorical_features = config["preprocessing"]["categorical_features"]

    num_cfg = config["preprocessing"]["numerical_pipeline"]
    cat_cfg = config["preprocessing"]["categorical_pipeline"]

    scaler_class = SCALERS[num_cfg["scaler"]]
    encoder_class = ENCODERS[cat_cfg["encoder"]]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=config["preprocessing"]["numerical_pipeline"]["imputer_strategy"])),
            ("scaler", scaler_class()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=config["preprocessing"]["categorical_pipeline"]["imputer_strategy"], fill_value="missing")),
            ("encoder", encoder_class(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor
