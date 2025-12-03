import pandas as pd
import numpy as np
import joblib
import os

from housing.preprocessing.preprocessing import preprocess
from housing.features.create_features import add_basic_features
from housing.features.geo_features import add_geo_bins, target_encode_apply
from housing.config import (
    TEST_DATA_PATH,
    RAW_TARGET,
    LOG_TARGET,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    GEO_BIN
)


###################
### PREDICT MODEL
###################

def predict_model():

    print("✅Loading test data...")
    df = pd.read_csv(TEST_DATA_PATH)

    print("✅Applying preprocessing...")
    df = preprocess(df, is_train=False)

    print("✅Adding basic engineered features...")
    df = add_basic_features(df)

    print("✅Adding geo bins...")
    df = add_geo_bins(df)

    print("✅Loading saved geo-bin target encoding mapping...")
    mapping = joblib.load("models/geo_bin_te_mapping.pkl")

    print("✅Applying target encoding to geo bins...")
    df["geo_bin_te"] = target_encode_apply(
        df,
        col=GEO_BIN,
        mapping=mapping
    )

    #######################
    ### FINAL FEATURE SET
    #######################

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [GEO_BIN]
    X_test = df[feature_cols]

    print(f"✅Final test matrix shape: {X_test.shape}")

    print("✅Loading trained model pipeline...")
    model = joblib.load("models/regression_pipeline.pkl")

    print("✅Predicting log target...")
    y_log_pred = model.predict(X_test)

    print("✅Inverting log-transform...")
    y_pred = np.expm1(y_log_pred)

    print("✅Saving predictions to CSV...")
    submission = pd.DataFrame({
        "lotid": df["lotid"],  # ensure exists in test data
        RAW_TARGET: y_pred
    })

    os.makedirs("predictions", exist_ok=True)
    submission.to_csv("predictions/predictions.csv", index=False)

    print("✅Prediction complete! Saved to predictions/predictions.csv")
    return submission


##############################
### Script entrypoint
##############################

if __name__ == "__main__":
    predict_model()
