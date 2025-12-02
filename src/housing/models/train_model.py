import pandas as pd
import joblib
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error

from src.preprocessing.preprocessing import preprocess
from src.features.create_features import add_basic_features
from src.features.geo_features import oof_target_encode_train, add_geo_bins
from src.config import (
    RAW_DATA_PATH,
    # MODEL_PATH, 
    # ENCODER_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    POLY_FEATURES,
    N_SPLITS,
    RIDGE_PARAMS,
    RAW_TARGET,
    LOG_TARGET,
    GEO_BIN
)
# import importlib
# import src.config
# importlib.reload(src.config)

##################
### TRAIN MODEL ###
##################

def train_model():
    
    print("✅Loading training data")
    df = pd.read_csv(RAW_DATA_PATH)

    print("✅Transform target feature")
    df[LOG_TARGET] = np.log1p(df[RAW_TARGET])

    print("✅Preprocessing")
    df = preprocess(df, is_train=True)

    print("✅Creating new features")
    df = add_basic_features(df)

    print("✅Creating geo features")
    df = add_geo_bins(df)

    # Create geo bins
    df["geo_bin_te"], mapping = oof_target_encode_train(df, col = GEO_BIN, target= LOG_TARGET, n_splits = N_SPLITS)

    print("✅Final X with features ready")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [GEO_BIN] ]
    y = df[LOG_TARGET]


    # Transformers
    ##############

    log_transformer = FunctionTransformer(np.log1p, feature_names_out="one-to-one")

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log", log_transformer),
        ("scaler", StandardScaler())
    ])

    poly_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log", log_transformer),
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("poly", poly_transformer, POLY_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    ##################
    ### TRAIN MODEL ###
    ##################

    print("✅Trained model: Ridge Regression ")
    model = Ridge(**RIDGE_PARAMS)

    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    full_pipeline.fit(X, y)

    # Check model error
    y_hat_in = full_pipeline.predict(X)
    print(root_mean_squared_error(y_hat_in, y))

    print("✅Saving full model pipeline")
    joblib.dump(full_pipeline, "models/regression_pipeline.pkl")

    print("✅Saving geo_bin_te encoding mapping")
    joblib.dump(mapping, "models/geo_bin_te_mapping.pkl")

    print("✅Training completed")
    return full_pipeline



if __name__ == "__main__":
    train_model()