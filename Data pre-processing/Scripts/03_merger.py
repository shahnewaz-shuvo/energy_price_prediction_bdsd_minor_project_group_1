# Script 03 - merger
# joins all the parsed parquet tables into one wide hourly table, using the


import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_regimeB")


def load_parquet(name):
    # load a parquet by name, or None if it's missing
    path = os.path.join(OUTPUT_DIR, f"{name}.parquet")
    if not os.path.exists(path):
        print(f"  [SKIP] {name}.parquet not found")
        return None
    df = pd.read_parquet(path)
    print(f"  [OK]   {name}.parquet - {len(df):,} rows, {len(df.columns)} columns")
    return df


def to_hourly(df, numeric_agg="mean"):
    # round timestamps to the hour and average numeric columns (some files have
    # 15-min data or multiple locations per hour). keep source_month / is_forecast_month.
    if df is None or df.empty:
        return df

    df = df.copy()
    df["timestamp"] = df["timestamp"].dt.floor("h")

    meta_cols = [c for c in df.columns if c in ("timestamp", "source_month",
                                                  "is_forecast_month", "location")]
    numeric_cols = [c for c in df.columns if c not in meta_cols]

    agg_dict = {}
    for col in numeric_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_dict[col] = numeric_agg
        else:
            agg_dict[col] = "first"

    if "source_month" in df.columns:
        agg_dict["source_month"] = "first"
    if "is_forecast_month" in df.columns:
        agg_dict["is_forecast_month"] = "first"

    if "location" in agg_dict:
        del agg_dict["location"]

    df = df.drop(columns=[c for c in ["location"] if c in df.columns])
    df = df.groupby("timestamp", as_index=False).agg(agg_dict)
    return df


print("=" * 70)
print("SCRIPT 03 - MERGER")
print("=" * 70)
print(f"Output folder: {OUTPUT_DIR}")
print()

# load everything
print("STEP 1 - Loading parquet files")
print("-" * 50)

energy         = load_parquet("parsed_energy")
weather        = load_parquet("parsed_weather")
wind           = load_parquet("parsed_wind")
solar          = load_parquet("parsed_solar")
demand_weather = load_parquet("parsed_demand_weather")
calendar       = load_parquet("parsed_calendar")
cross_border   = load_parquet("parsed_cross_border")
load_fc        = load_parquet("parsed_load")
generation     = load_parquet("parsed_generation")
market_proxies = load_parquet("parsed_market_proxies")
ned_production = load_parquet("parsed_ned_production")
grid_imbalance = load_parquet("parsed_grid_imbalance")
gas_flows      = load_parquet("parsed_gas_flows")
gas_storage    = load_parquet("parsed_gas_storage")

if energy is None or energy.empty:
    print("\nFATAL: parsed_energy.parquet is missing or empty. Run Script 02 first.")
    exit(1)

# round everything to hourly
print("\nSTEP 2 - Rounding to hourly")
print("-" * 50)

energy         = to_hourly(energy)
weather        = to_hourly(weather)
wind           = to_hourly(wind)
solar          = to_hourly(solar)
demand_weather = to_hourly(demand_weather)
calendar       = to_hourly(calendar)
cross_border   = to_hourly(cross_border)
load_fc        = to_hourly(load_fc)
generation     = to_hourly(generation)
market_proxies = to_hourly(market_proxies)
ned_production = to_hourly(ned_production)
grid_imbalance = to_hourly(grid_imbalance)
gas_flows      = to_hourly(gas_flows)
gas_storage    = to_hourly(gas_storage)

print("  done.")

# left join everything onto the energy table
print("\nSTEP 3 - Joining onto the energy table")
print("-" * 50)

# drop the meta columns from the other tables so they don't clash
def drop_meta(df, keep_ts=True):
    if df is None:
        return None
    drop_cols = ["source_month", "is_forecast_month"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df

merged = energy.copy()
print(f"  anchor (energy): {len(merged):,} rows")

tables_to_join = [
    ("weather",        drop_meta(weather)),
    ("wind",           drop_meta(wind)),
    ("solar",          drop_meta(solar)),
    ("demand_weather", drop_meta(demand_weather)),
    ("calendar",       drop_meta(calendar)),
    ("cross_border",   drop_meta(cross_border)),
    ("load_forecast",  drop_meta(load_fc)),
    ("generation",     drop_meta(generation)),
    ("market_proxies", drop_meta(market_proxies)),
    ("ned_production", drop_meta(ned_production)),
    ("grid_imbalance", drop_meta(grid_imbalance)),
    ("gas_flows",      drop_meta(gas_flows)),
    ("gas_storage",    drop_meta(gas_storage)),
]

for name, df in tables_to_join:
    if df is None or df.empty:
        print(f"  [{name}] not available, skipping")
        continue

    # if a column name clashes, add a suffix
    overlap = [c for c in df.columns if c in merged.columns and c != "timestamp"]
    if overlap:
        df = df.rename(columns={c: f"{c}_{name}" for c in overlap})

    merged = pd.merge(merged, df, on="timestamp", how="left")
    print(f"  joined [{name}] - now {len(merged):,} rows x {len(merged.columns)} columns")

# drop rows that have no price (can't train on them)
print("\nSTEP 4 - Dropping rows without ENTSOE price")
print("-" * 50)

rows_before = len(merged)
merged = merged.dropna(subset=["price_entsoe"])
rows_after = len(merged)
print(f"  rows before : {rows_before:,}")
print(f"  rows dropped: {rows_before - rows_after:,}")
print(f"  rows left   : {rows_after:,}")

merged = merged.sort_values("timestamp").reset_index(drop=True)

# summary
print("\nSTEP 5 - Summary")
print("-" * 50)
print(f"  total rows   : {len(merged):,}")
print(f"  total columns: {len(merged.columns)}")
print(f"  date range   : {merged['timestamp'].min()} -> {merged['timestamp'].max()}")
print(f"\n  columns and how full they are:")
for i, col in enumerate(merged.columns):
    non_null = merged[col].notna().sum()
    pct = 100 * non_null / len(merged)
    print(f"    {i+1:>3}. {col:<45} {non_null:>6,} / {len(merged):,} ({pct:.1f}% filled)")

# how many measured vs forecast rows
print("\n  measured vs forecast rows:")
if "is_forecast_month" in merged.columns:
    counts = merged["is_forecast_month"].value_counts()
    for val, count in counts.items():
        label = "forecast (Mar-May 2026)" if val else "measured (Dec 2025-Feb 2026)"
        print(f"    {label}: {count:,} rows")

# save
out_path = os.path.join(OUTPUT_DIR, "merged_regimeB.parquet")
merged.to_parquet(out_path, index=False)
print(f"\n  saved to: {out_path}")

print("\n" + "=" * 70)
print("=" * 70)
