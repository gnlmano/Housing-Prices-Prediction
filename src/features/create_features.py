import numpy as np
# Function to create the fetures required
def add_basic_features(df):
    df = df.copy()
    df['avg_room_size'] = df['finishedarea'] / df['roomnum'].replace(0, np.nan)
    df['bed_bath_ratio'] = df['numbedroom'] / (df['numfullbath'] + df['num34bath']).replace(0, np.nan)
    # Correct the following features
    df['roomnum'] = df['roomnum'].replace(0, np.nan)
    df['lotarea'] = df['lotarea'].replace(0, np.nan)
    df['lotarea_log'] = np.log1p(df['lotarea'])
    df['lotarea_log'] = df['lotarea_log'].fillna(0)
    return df


