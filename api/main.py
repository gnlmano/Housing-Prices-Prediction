from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel
from typing import Optional
import json
import random

# Import your preprocessing + feature engineering
from src.preprocessing.preprocessing import preprocess
from src.features.create_features import add_basic_features
from src.features.geo_features import add_geo_bins, target_encode_apply
from src.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    GEO_BIN
)

app = FastAPI(
    title="Housing Price Prediction API",
    description="Predicts parcel values using the engineered Ridge Regression pipeline.",
    version="1.0",
)

# Load trained pipeline + mapping
MODEL_PATH = "models/regression_pipeline.pkl"
MAPPING_PATH = "models/geo_bin_te_mapping.pkl"

model = joblib.load(MODEL_PATH)
geo_mapping = joblib.load(MAPPING_PATH)

# Load test samples
with open("api/sample_rows.json", "r") as f:
    SAMPLE_ROWS = json.load(f)

@app.get("/sample")
def get_sample():
    """
    Returns a random valid test row from the real test dataset.
    Safe to submit directly to /predict.
    """
    return random.choice(SAMPLE_ROWS)

# -----------------------------
# Pydantic input schema
# -----------------------------
class HouseRow(BaseModel):
    aircond: int | None = None
    qualitybuild: int | None = None
    heatingtype: int | None = None
    unitnum: int | None = None
    basement: Optional[float] = None
    numbedroom: Optional[float] = None
    decktype: Optional[float] = None
    finishedarea: float
    finishedareaEntry: Optional[float] = None
    countycode: Optional[int] = None
    numfireplace: Optional[float] = None
    numfullbath: Optional[float] = None
    garagenum: Optional[float] = None
    garagearea: Optional[float] = None
    tubflag: Optional[bool] = None
    latitude: float
    longitude: float
    lotarea: Optional[float] = None
    poolnum: Optional[float] = None
    poolarea: Optional[float] = None
    citycode: Optional[int] = None
    countycode2: Optional[int] = None
    neighborhoodcode: Optional[int] = None
    regioncode: int
    roomnum: Optional[float] = None
    num34bath: Optional[float] = None
    year: Optional[float] = None
    numstories: Optional[float] = None
    taxyear: Optional[float] = None
    taxdelinquencyflag: Optional[str] = None
    taxdelinquencyyear: Optional[float] = None

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_house_price(row: HouseRow):

    # Convert incoming JSON → DataFrame
    df = pd.DataFrame([row.dict()])

    # 1. PREPROCESS (drop, impute basic raw data)
    df = preprocess(df, is_train=False)

    # 2. ENGINEER BASIC FEATURES (avg_room_size, lotarea_log, ratios, etc.)
    df = add_basic_features(df)

    # 3. GEO BINS (lat-lon grid feature)
    df = add_geo_bins(df)

    # 4. TARGET ENCODING (geo_bin_te)
    df["geo_bin_te"] = target_encode_apply(
        df, col=GEO_BIN, mapping=geo_mapping
    )

    # 5. FINAL FEATURE SET
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [GEO_BIN]]

    # 6. Predict log target
    y_log = model.predict(X)[0]

    # 7. Invert log1p → actual parcel value
    y_pred = float(np.expm1(y_log))

    return {
        "prediction": y_pred,
        "note": "Value is in original parcel-value scale."
    }


@app.get("/")
def root():
    return {"message": "Housing ML API is running. Visit /docs to test."}
