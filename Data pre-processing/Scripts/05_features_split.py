# Script 05 - feature engineering + split
# takes the clean csv and makes the model-ready files for FFNN, LSTM and Bi-LSTM.


import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# paths
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_regimeB")
INPUT_CSV  = os.path.join(OUTPUT_DIR, "regime_B_clean.csv")
ML_DIR     = os.path.join(OUTPUT_DIR, "ml_ready")
os.makedirs(ML_DIR, exist_ok=True)

# settings
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# test = the rest (~15%)

HORIZON    = 1     # predict 1 hour ahead, then run recursively for the 72h forecast
TIME_STEPS = 24    # how many past hours the LSTM looks at

# True = train only on the measured months, False = use all rows
USE_ONLY_MEASURED = False

print("=" * 65)
print("SCRIPT 05 - FEATURE ENGINEERING + SPLIT")
print("=" * 65)
print(f"Input : {INPUT_CSV}")
print(f"Output: {ML_DIR}")
print()

if not os.path.exists(INPUT_CSV):
    print("ERROR: regime_B_clean.csv not found. Run Script 04 first.")
    exit()

# load
print("STEP 1 - Loading clean dataset")
print("-" * 50)
df = pd.read_csv(INPUT_CSV)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"  Loaded: {len(df):,} rows x {df.shape[1]} columns")
print(f"  Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
print(f"  measured  : {(df['data_type']=='measured').sum():,} rows")
print(f"  predicted : {(df['data_type']=='predicted').sum():,} rows")
print()

# optional: keep only measured months
if USE_ONLY_MEASURED:
    before = len(df)
    df = df[df['data_type'] == 'measured'].copy().reset_index(drop=True)
    print("STEP 2 - Keeping measured data only")
    print("-" * 50)
    print(f"  Rows before: {before:,}")
    print(f"  Rows after : {len(df):,}")
    print()
else:
    print("STEP 2 - Keeping all data (measured + predicted)")
    print("-" * 50)
    print()

# sin/cos time features
print("STEP 3 - Cyclical time features (sin/cos)")
print("-" * 50)
h   = df['cal_hour']
dow = df['cal_day_of_week']
mon = df['cal_month']

df['hour_sin']  = np.sin(2 * np.pi * h   / 24)
df['hour_cos']  = np.cos(2 * np.pi * h   / 24)
df['dow_sin']   = np.sin(2 * np.pi * dow / 7)
df['dow_cos']   = np.cos(2 * np.pi * dow / 7)
df['month_sin'] = np.sin(2 * np.pi * mon / 12)
df['month_cos'] = np.cos(2 * np.pi * mon / 12)
print("  added hour/day/month sin+cos")
print()

# price lags
print("STEP 4 - Price lag features")
print("-" * 50)
df['price_lag_1h']   = df['price_entsoe'].shift(1)
df['price_lag_24h']  = df['price_entsoe'].shift(24)
df['price_lag_168h'] = df['price_entsoe'].shift(168)
print("  added 1h, 24h, 168h lags")
print()

# rolling stats
print("STEP 5 - 24h rolling stats")
print("-" * 50)
df['price_roll24_mean'] = df['price_entsoe'].rolling(24).mean()
df['price_roll24_std']  = df['price_entsoe'].rolling(24).std()
print("  added 24h mean and std")
print()

# target = next hour price
print("STEP 6 - Target (next hour)")
print("-" * 50)
df['target'] = df['price_entsoe'].shift(-HORIZON)
print(f"  target = price shifted by -{HORIZON}")
print()

# drop the rows that became NaN at the edges (from lags / rolling / target)
print("STEP 7 - Dropping NaN edge rows")
print("-" * 50)
before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"  Rows before: {before:,}")
print(f"  Rows after : {len(df):,}")
print(f"  Dropped    : {before - len(df):,}")
print()

# pick which columns go into the model
print("STEP 8 - Feature columns")
print("-" * 50)

# things we don't feed to the model
NON_FEATURE_COLS = [
    'timestamp', 'source_month', 'data_type',
    'cal_hour', 'cal_day_of_week', 'cal_month',  # replaced by the sin/cos versions
    'target',
]

feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
print(f"  {len(feature_cols)} feature columns:")
for c in feature_cols:
    print(f"    - {c}")
print()

# split by time (no shuffle - it's a time series)
print("STEP 9 - Chronological 70/15/15 split")
print("-" * 50)
n = len(df)
n_train = int(n * TRAIN_FRAC)
n_val   = int(n * VAL_FRAC)

train = df.iloc[:n_train].copy()
val   = df.iloc[n_train:n_train + n_val].copy()
test  = df.iloc[n_train + n_val:].copy()

print(f"  train: {len(train):>5,} rows  {train['timestamp'].min()} -> {train['timestamp'].max()}")
print(f"  val  : {len(val):>5,} rows  {val['timestamp'].min()} -> {val['timestamp'].max()}")
print(f"  test : {len(test):>5,} rows  {test['timestamp'].min()} -> {test['timestamp'].max()}")
print()

# split into X and y
print("STEP 10 - X / y")
print("-" * 50)
X_train = train[feature_cols]
X_val   = val[feature_cols]
X_test  = test[feature_cols]

y_train = train['target']
y_val   = val['target']
y_test  = test['target']

print(f"  X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")
print()

# scale - fit on train only so val/test don't leak into it
print("STEP 11 - Scaling (fit on train only)")
print("-" * 50)
scaler_X = StandardScaler()
X_train_sc = pd.DataFrame(scaler_X.fit_transform(X_train),
                          index=X_train.index, columns=X_train.columns)
X_val_sc   = pd.DataFrame(scaler_X.transform(X_val),
                          index=X_val.index, columns=X_val.columns)
X_test_sc  = pd.DataFrame(scaler_X.transform(X_test),
                          index=X_test.index, columns=X_test.columns)

scaler_y = StandardScaler()
y_train_sc = pd.Series(scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
                       index=y_train.index, name='target')
y_val_sc   = pd.Series(scaler_y.transform(y_val.values.reshape(-1, 1)).ravel(),
                       index=y_val.index, name='target')
y_test_sc  = pd.Series(scaler_y.transform(y_test.values.reshape(-1, 1)).ravel(),
                       index=y_test.index, name='target')
print("  fitted on train, applied to val and test")
print()

# 2D tables for the FFNN
print("STEP 12 - 2D tables for FFNN")
print("-" * 50)
train_out = X_train_sc.copy(); train_out['target'] = y_train_sc
val_out   = X_val_sc.copy();   val_out['target']   = y_val_sc
test_out  = X_test_sc.copy();  test_out['target']  = y_test_sc
print(f"  train: {train_out.shape} | val: {val_out.shape} | test: {test_out.shape}")
print()

# 3D sequences for LSTM / Bi-LSTM
print(f"STEP 13 - 3D sequences for LSTM (TIME_STEPS = {TIME_STEPS})")
print("-" * 50)

def make_sequences(X, y, time_steps):
    # slide a window over the rows -> (samples, time_steps, n_features)
    X_seq, y_seq = [], []
    for i in range(time_steps, len(X)):
        X_seq.append(X.iloc[i - time_steps:i].values)
        y_seq.append(y.iloc[i])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = make_sequences(X_train_sc, y_train_sc, TIME_STEPS)
X_val_seq,   y_val_seq   = make_sequences(X_val_sc,   y_val_sc,   TIME_STEPS)
X_test_seq,  y_test_seq  = make_sequences(X_test_sc,  y_test_sc,  TIME_STEPS)

print(f"  X_train_seq: {X_train_seq.shape}  y_train_seq: {y_train_seq.shape}")
print(f"  X_val_seq  : {X_val_seq.shape}  y_val_seq  : {y_val_seq.shape}")
print(f"  X_test_seq : {X_test_seq.shape}  y_test_seq : {y_test_seq.shape}")
print()

# save everything
print("STEP 14 - Saving")
print("-" * 50)

# FFNN tables
train_out.to_parquet(os.path.join(ML_DIR, "train.parquet"))
val_out.to_parquet  (os.path.join(ML_DIR, "val.parquet"))
test_out.to_parquet (os.path.join(ML_DIR, "test.parquet"))
print("  saved train/val/test.parquet")

# un-scaled csvs to look at
train.to_csv(os.path.join(ML_DIR, "train_raw.csv"), index=False)
val.to_csv  (os.path.join(ML_DIR, "val_raw.csv"),   index=False)
test.to_csv (os.path.join(ML_DIR, "test_raw.csv"),  index=False)
print("  saved train/val/test_raw.csv (un-scaled)")

# LSTM sequences
np.save(os.path.join(ML_DIR, "X_train_seq.npy"), X_train_seq)
np.save(os.path.join(ML_DIR, "y_train_seq.npy"), y_train_seq)
np.save(os.path.join(ML_DIR, "X_val_seq.npy"),   X_val_seq)
np.save(os.path.join(ML_DIR, "y_val_seq.npy"),   y_val_seq)
np.save(os.path.join(ML_DIR, "X_test_seq.npy"),  X_test_seq)
np.save(os.path.join(ML_DIR, "y_test_seq.npy"),  y_test_seq)
print("  saved X_*_seq.npy and y_*_seq.npy")

# scalers
with open(os.path.join(ML_DIR, "scaler_X.pkl"), "wb") as f:
    pickle.dump(scaler_X, f)
with open(os.path.join(ML_DIR, "scaler_y.pkl"), "wb") as f:
    pickle.dump(scaler_y, f)
print("  saved scaler_X.pkl and scaler_y.pkl")
print()

print("=" * 65)
print("DONE")
print("=" * 65)
print("FFNN  : train/val/test.parquet")
print("LSTM  : X_*_seq.npy + y_*_seq.npy")
print("BiLSTM: same as LSTM")
print()
print("notes for the model team:")
print("  - use scaler_y.inverse_transform(pred) to get back EUR/MWh")
print("  - for the 72h forecast, run the model 72 times, feeding each")
print("    prediction back in as the next input")
print("=" * 65)
