import numpy as np
from sklearn.model_selection import KFold
from housing.config import (
    LAT_BIN_SCALE, LON_BIN_SCALE, LOG_TARGET, N_SPLITS, TARGET_ENCODER_K
)

# Function to create bins around the lat-lon
def add_geo_bins(df):
    df = df.copy()
    df['lat_bin'] = (df['latitude'] * LAT_BIN_SCALE).round().astype(int)
    df['lon_bin'] = (df['longitude'] * LON_BIN_SCALE).round().astype(int)
    df['geo_bin'] = df['lat_bin'].astype(str) + "_" + df['lon_bin'].astype(str)
    return df

def oof_target_encode_train(train_df, col, target, n_splits = N_SPLITS, k=TARGET_ENCODER_K, random_state=42):
    train_encoded = np.zeros(len(train_df))
    global_mean = train_df[target].mean()

    kf = KFold(n_splits = n_splits, shuffle=True, random_state=random_state)

    for tr_idx, val_idx in kf.split(train_df):
        tr = train_df.iloc[tr_idx]
        val = train_df.iloc[val_idx]

        stats = tr.groupby(col)[target].agg(['mean', 'count'])
        smooth = (stats['count'] * stats['mean'] + k * global_mean) / (stats['count'] + k)

        train_encoded[val_idx] = val[col].map(smooth).fillna(global_mean)

    # learn final mapping for test
    full_stats = train_df.groupby(col)[target].agg(['mean', 'count'])
    mapping = (full_stats['count'] * full_stats['mean'] + k * global_mean) / (full_stats['count'] + k)

    return train_encoded, mapping

def target_encode_apply(df, col, mapping, global_mean=None):
    if global_mean is None:
        global_mean = mapping.mean()

    return df[col].map(mapping).fillna(global_mean)