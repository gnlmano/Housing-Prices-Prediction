# PATHS
RAW_DATA_PATH = "data/Regression_Supervised_Train.csv"
TEST_DATA_PATH = "data/Regression_Supervised_Test.csv"
### MODEL PARAMETERS
RIDGE_PARAMS = {"alpha": 1}
# FEATURES TO DROP AS PER INSTRUCTIONS
DROP_TRAIN = ['buildvalue', 'landvalue', 'mypointer', 'logerror', 'totaltaxvalue', 'lotid']
DROP_TEST = ['buildvalue', 'landvalue', 'logerror', 'totaltaxvalue']
# FEATURE NAMES
CATEGORICAL_FEATURES= ['aircond', 'qualitybuild', 'decktype', 'tubflag', 'heatingtype',
               'taxdelinquencyflag', 'unitnum', 'regioncode']

NUMERIC_FEATURES =   ['basement', 'numbedroom', 'finishedarea',
               'finishedareaEntry', 'numfireplace', 'numfullbath', 'garagenum', 
               'garagearea', 'lotarea_log', 'poolnum', 'poolarea', 'roomnum', 
               'num34bath', 'year', 'numstories', 'taxyear',
               'taxdelinquencyyear', 'avg_room_size', 'bed_bath_ratio', 'geo_bin_te']
POLY_FEATURES = ['finishedarea', 'finishedareaEntry' ]

RAW_TARGET = "parcelvalue"
LOG_TARGET = "log_parcelvalue"
GEO_BIN = "geo_bin"
ENGINEERED_FEATURES = ['avg_room_size', 'bed_bath_ratio', 'geo_bin_te']
### Preprocessing settings
PRESENT_YEAR = 2020
LAT_BIN_SCALE = 50
LON_BIN_SCALE = 50
IMPUTE_ZERO = ['basement',  'decktype', 'num34bath', 'garagearea', 'poolarea', 'taxdelinquencyyear', 'garagenum', 'poolnum']
IMPUTE_FALSE = ['tubflag'] # 
IMPUTE_NO = ['taxdelinquencyflag']
PRESENT_YEAR = 2020
N_SPLITS = 5
RANDOM_STATE = 42
### Target Encoding
TARGET_ENCODER_K = 20