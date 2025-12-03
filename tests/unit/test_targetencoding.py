import pandas as pd
from housing.features.geo_features import oof_target_encode_train

def test_oof_encoding_shapes():
    df = pd.DataFrame({
        "geo_bin": ["A", "A", "B", "B", "C"],
        "log_parcelvalue": [10, 12, 20, 18, 30]
    })

    encoded, mapping = oof_target_encode_train(df, "geo_bin", "log_parcelvalue", n_splits=2)

    # encoded length equal to df lenght
    assert len(encoded) == len(df)

    # mapping should contain the distinct categories
    assert set(mapping.index) == {"A", "B", "C"}

    