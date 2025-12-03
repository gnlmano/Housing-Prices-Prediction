import pandas as pd
from housing.preprocessing.preprocessing import preprocess, drop_columns, impute_missing
from housing.config import DROP_TRAIN, IMPUTE_ZERO, IMPUTE_FALSE, IMPUTE_NO, PRESENT_YEAR

# Test drop columns
def test_drop_columns_train():
    df = pd.DataFrame({
        "buildvalue": [1],
        "landvalue": [1],
        "aircond": ["A"],
    })

    processed = drop_columns(df.copy(), is_train=True)

    for col in DROP_TRAIN:
        assert col not in processed.columns, "Columns Dropping Failed"

# Test imputing works
def test_impute_missing():
    df = pd.DataFrame({
        "basement": [None],
        "decktype": [None],
        "tubflag": [None],
        "num34bath": [None],
        "garagearea": [None],
        "poolarea": [None],
        "taxdelinquencyyear": [None],
        "garagenum": [None],
        "poolnum": [None],
        "taxdelinquencyflag": [None],
        "year": [2000],
        "taxyear": [2010],
    })

    processed = impute_missing(df.copy())

    # ZERO imputation
    for col in IMPUTE_ZERO:
        if col in df.columns:
            assert processed[col].iloc[0] == 0, "Impute Zero Failed"

    # FALSE imputation
    for col in IMPUTE_FALSE:
        if col in df.columns:
            assert processed[col].iloc[0] == False, "Impute False Failed"

    # "N" imputation
    for col in IMPUTE_NO:
        if col in df.columns:
            assert processed[col].iloc[0] == "N", "Impute No Failed"

    # Year transformations
    assert processed["year"].iloc[0] == PRESENT_YEAR - 2000,  "Year Transformation Failed"
    assert processed["taxyear"].iloc[0] == PRESENT_YEAR - 2010,  "Tax Year Transformation Failed"

# Test full preprocessor:
def test_preprocess_end_to_end():
    df = pd.DataFrame({
        "basement": [None],
        "decktype": [None],
        "tubflag": [None],
        "num34bath": [None],
        "garagearea": [None],
        "poolarea": [None],
        "taxdelinquencyyear": [None],
        "garagenum": [None],
        "poolnum": [None],
        "taxdelinquencyflag": [None],
        "year": [2000],
        "taxyear": [2010],
        "buildvalue": [1],
        "landvalue": [1],
        "aircond": ["A"],
    })
    
    processed = preprocess(df.copy(), is_train=True)

     # dropped columns
    assert "buildvalue" not in processed.columns, "Columns Dropping (buildvalue) Failed"
    assert "landvalue" not in processed.columns, "Columns Dropping (landvalue) Failed"


    for col in IMPUTE_FALSE:
        if col in df.columns:
            assert processed[col].iloc[0] == False, "Impute False Failed"

    # "N" imputation
    for col in IMPUTE_NO:
        if col in df.columns:
            assert processed[col].iloc[0] == "N", "Impute No Failed"

    # Year transformations
    assert processed["year"].iloc[0] == PRESENT_YEAR - 2000,  "Year Transformation Failed"
    assert processed["taxyear"].iloc[0] == PRESENT_YEAR - 2010,  "Tax Year Transformation Failed"