import pandas as pd
from src.features.geo_features import add_geo_bins
from src.config import LAT_BIN_SCALE, LON_BIN_SCALE

def test_geo_bin_creation():
    df = pd.DataFrame({
        "latitude": [34.05],
        "longitude": [-118.25]
    })

    out = add_geo_bins(df)

    expected_lat = int(round(34.05 * LAT_BIN_SCALE))
    expected_lon = int(round(-118.25 * LON_BIN_SCALE))
    expected_string = f"{expected_lat}_{expected_lon}"

    assert "geo_bin" in out.columns, "Adding geobin failed"
    assert out["geo_bin"].iloc[0] == expected_string, "Geobin expected value failed"