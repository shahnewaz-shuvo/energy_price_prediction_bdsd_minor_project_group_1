# Script 04 - cleaner
# takes the merged table from script 03 and keeps only the columns we want,

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# paths
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_regimeB")
INPUT_FILE = os.path.join(OUTPUT_DIR, "merged_regimeB.parquet")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "regime_B_clean.csv")

# the columns we keep: meta + price target + 18 features
SELECTED_COLUMNS = [
    # meta (reference only, not model inputs)
    'timestamp',
    'source_month',
    'is_forecast_month',
    'data_type',

    # target
    'price_entsoe',

    # solar
    'solar_ghi',
    'solar_cloud_cover',

    # wind (both m/s)
    'wind_speed_ms',
    'dw_wind_speed',

    # weather / demand
    'dw_temperature',
    'dw_humidity',
    'dw_hdd',                       # heating degree days

    # load forecasts (forecast values, so no leakage)
    'load_NL_load_forecast_mw',
    'load_DE_LU_load_forecast_mw',

    # cross-border flows (kept the two that correlate with price)
    'flow_NL_NO_net_mw',
    'flow_NL_GB_net_mw',

    # french nuclear
    'gen_FR_nuclear_actual_mw',

    # grid balance
    'grid_balance_delta_mw',

    # calendar
    'cal_hour',
    'cal_day_of_week',
    'cal_month',
    'cal_is_working_day',
    'cal_is_holiday_nl',
]

print("=" * 65)
print("SCRIPT 04 - REGIME B CLEANER")
print("=" * 65)
print(f"Input : {INPUT_FILE}")
print(f"Output: {OUTPUT_CSV}")
print()

if not os.path.exists(INPUT_FILE):
    print("ERROR: merged_regimeB.parquet not found. Run Script 03 first.")
    exit()

# load merged table
print("STEP 1 - Loading merged table")
print("-" * 50)
df = pd.read_parquet(INPUT_FILE)
print(f"  Loaded: {len(df):,} rows x {df.shape[1]} columns")
print()

# label each row measured / predicted
print("STEP 2 - Adding data_type column")
print("-" * 50)
df['data_type'] = df['is_forecast_month'].map({True: 'predicted', False: 'measured'})
print(f"  measured  : {(df['data_type']=='measured').sum():,} rows")
print(f"  predicted : {(df['data_type']=='predicted').sum():,} rows")
print()

# make sure the columns we asked for are actually there
print("STEP 3 - Checking selected columns exist")
print("-" * 50)
missing_cols = [c for c in SELECTED_COLUMNS if c not in df.columns]
if missing_cols:
    print(f"  WARNING: these selected columns were not found:")
    for c in missing_cols:
        print(f"    - {c}")
    SELECTED_COLUMNS_FINAL = [c for c in SELECTED_COLUMNS if c in df.columns]
else:
    print(f"  All {len(SELECTED_COLUMNS)} columns found.")
    SELECTED_COLUMNS_FINAL = SELECTED_COLUMNS
print()

# keep only those columns
print("STEP 4 - Selecting features")
print("-" * 50)
df_sel = df[SELECTED_COLUMNS_FINAL].copy()
print(f"  Columns before : {df.shape[1]}")
print(f"  Columns after  : {df_sel.shape[1]}")
print(f"  Columns dropped: {df.shape[1] - df_sel.shape[1]}")
print()

# fill the leftover gaps
print("STEP 5 - Filling missing values")
print("-" * 50)
before = df_sel.isnull().sum().sum()
print(f"  Missing cells before: {before:,}")

df_sel = df_sel.sort_values('timestamp').reset_index(drop=True)

# forward fill then backward fill (price has no gaps here, it was dropped earlier if missing)
numeric_cols = df_sel.select_dtypes(include=[np.number]).columns.tolist()
df_sel[numeric_cols] = df_sel[numeric_cols].ffill().bfill()

after = df_sel.isnull().sum().sum()
print(f"  Missing cells after : {after:,}")
print()

# quick summary
print("STEP 6 - Final dataset summary")
print("-" * 50)
print(f"  Rows       : {len(df_sel):,}")
print(f"  Columns    : {df_sel.shape[1]}")
print(f"  Missing    : {df_sel.isnull().sum().sum()}")
print(f"  Duplicates : {df_sel['timestamp'].duplicated().sum()}")
print()

ts = pd.to_datetime(df_sel['timestamp'], utc=True)
print(f"  Date range : {ts.min()} -> {ts.max()}")
print()

measured  = (df_sel['data_type'] == 'measured').sum()
predicted = (df_sel['data_type'] == 'predicted').sum()
print(f"  Measured rows  : {measured:,}")
print(f"  Predicted rows : {predicted:,}")
print()

print("  Column fill rates:")
for col in SELECTED_COLUMNS_FINAL:
    filled = df_sel[col].notna().sum()
    pct    = filled / len(df_sel) * 100
    print(f"    {col:<45} {filled:,} / {len(df_sel):,} ({pct:.1f}%)")
print()

# price stats
p = df_sel['price_entsoe']
print("  ENTSOE price statistics:")
print(f"    Mean     : {p.mean():.2f} EUR/MWh")
print(f"    Std      : {p.std():.2f} EUR/MWh")
print(f"    Min      : {p.min():.2f} EUR/MWh")
print(f"    Max      : {p.max():.2f} EUR/MWh")
print(f"    Negative : {(p < 0).sum()} hours")
print()

# just checking the wind columns are in m/s
print("  Wind speed ranges (should be m/s):")
print(f"    wind_speed_ms  : {df_sel['wind_speed_ms'].min():.2f} to {df_sel['wind_speed_ms'].max():.2f}")
print(f"    dw_wind_speed  : {df_sel['dw_wind_speed'].min():.2f} to {df_sel['dw_wind_speed'].max():.2f}")
print()

# save
print("STEP 7 - Saving")
print("-" * 50)
df_sel.to_csv(OUTPUT_CSV, index=False)
size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
print(f"  Saved: {OUTPUT_CSV}")
print(f"  Size : {size_mb:.1f} MB")
print()

print("=" * 65)
print(f"DONE - {len(df_sel):,} rows x {df_sel.shape[1]} columns saved")
print("=" * 65)
