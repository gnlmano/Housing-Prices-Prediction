import pandas as pd
import numpy as np
from src.features.create_features import add_basic_features
from src.config import DROP_TRAIN, IMPUTE_ZERO, IMPUTE_FALSE, IMPUTE_NO, PRESENT_YEAR

# Test drop columns
def test_create_features():
    df = pd.DataFrame({
        "finishedarea": [1],
        "roomnum": [3],
        "numbedroom": [3],
        "numfullbath": [4],
        "num34bath": [2],
        "lotarea": [1]
    })

    processed = add_basic_features(df.copy())

    
    assert np.isclose(processed['avg_room_size'].iloc[0], 1/3), "avg_room_size Failed"
    assert processed['bed_bath_ratio'].iloc[0] == 0.5, "bed_bath_ratio Failed"
    assert processed['lotarea_log'].iloc[0] == np.log(2), "lotarea_log Failed"