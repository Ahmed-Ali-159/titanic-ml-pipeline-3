from omegaconf import OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# from src.config import config


SCALERS = {
    "standard": StandardScaler,
}

ENCODERS = {
    "onehot": OneHotEncoder,
}


def build_preprocessor(cfg) -> ColumnTransformer:
    # numeric_features = config["preprocessing"]["numerical_features"]
    # categorical_features = config["preprocessing"]["categorical_features"]

    # numeric_features = cfg.preprocessing.numerical_features
    # categorical_features = cfg.preprocessing.categorical_features


    # OmegaConf lists are not plain Python lists, so sklearn's ColumnTransformer can't understand them. 
    # Fix it in preprocessor.py by converting them with OmegaConf.to_container():
    # Convert OmegaConf lists to plain Python lists
    numeric_features = OmegaConf.to_container(cfg.preprocessing.numerical_features)
    categorical_features = OmegaConf.to_container(cfg.preprocessing.categorical_features)


    # num_cfg = config["preprocessing"]["numerical_pipeline"]
    # cat_cfg = config["preprocessing"]["categorical_pipeline"]
    num_cfg = cfg.preprocessing.numerical_pipeline
    cat_cfg = cfg.preprocessing.categorical_pipeline

    scaler_class = SCALERS[num_cfg["scaler"]]
    encoder_class = ENCODERS[cat_cfg["encoder"]]

    numeric_pipeline = Pipeline(
        [
            # ("imputer", SimpleImputer(strategy=config["preprocessing"]["numerical_pipeline"]["imputer_strategy"])),
            ("imputer", SimpleImputer(strategy=num_cfg["imputer_strategy"])),
            ("scaler", scaler_class()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            # ("imputer", SimpleImputer(strategy=config["preprocessing"]["categorical_pipeline"]["imputer_strategy"], fill_value="missing")),
            ("imputer", SimpleImputer(strategy=cat_cfg["imputer_strategy"], fill_value="missing")),
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
