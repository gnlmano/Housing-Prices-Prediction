import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from housing.config import (
    DROP_TRAIN, DROP_TEST,
    IMPUTE_ZERO, IMPUTE_FALSE, IMPUTE_NO,
    PRESENT_YEAR,
    LAT_BIN_SCALE, LON_BIN_SCALE
)

# DROP FIXED COLUMNS
def drop_columns(df, is_train=True):
    df = df.copy()
    if is_train:
        df = df.drop(columns=DROP_TRAIN, errors="ignore")
    else:
        df = df.drop(columns=DROP_TEST, errors="ignore")
    return df

# IMPUTE THE 'LOGICAL' FEATURES
def impute_missing(df):
    df = df.copy()
    
    df[IMPUTE_ZERO] = df[IMPUTE_ZERO].fillna(0)
    df[IMPUTE_FALSE] = df[IMPUTE_FALSE].fillna(False)
    df[IMPUTE_NO] = df[IMPUTE_NO].fillna("N")
    
    # Year features — convert to durations
    df['year'] = df['year'].apply(lambda x: PRESENT_YEAR - x)
    df['taxyear'] = df['taxyear'].apply(lambda x: PRESENT_YEAR - x)
    
    return df


def preprocess(df, is_train=True):
    """
    Applies all preprocessing steps EXCEPT target encoding (because that depends on y).
    """
    df = drop_columns(df, is_train=is_train)
    df = impute_missing(df)
    return df