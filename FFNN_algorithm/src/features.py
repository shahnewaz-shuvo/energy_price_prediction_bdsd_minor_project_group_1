from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EXCLUDED_MODEL_COLUMNS, FINAL_TEST_ROWS, LOCAL_TZ, TARGET


def add_gap_safe_target_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    diffs = df["timestamp"].diff()
    df["continuous_segment_id"] = (diffs.ne(pd.Timedelta(hours=1))).cumsum()

    for lag in [96, 120, 168, 192, 336]:
        df[f"price_lag_{lag}h"] = df.groupby("continuous_segment_id", group_keys=False)[TARGET].shift(lag)

    grouped = df.groupby("continuous_segment_id", group_keys=False)[TARGET]
    shifted = grouped.shift(96)
    by_segment = shifted.groupby(df["continuous_segment_id"])
    df["price_roll_24_mean_shift96h"] = by_segment.rolling(24, min_periods=24).mean().reset_index(level=0, drop=True)
    df["price_roll_24_std_shift96h"] = by_segment.rolling(24, min_periods=24).std().reset_index(level=0, drop=True)
    df["price_roll_24_min_shift96h"] = by_segment.rolling(24, min_periods=24).min().reset_index(level=0, drop=True)
    df["price_roll_24_max_shift96h"] = by_segment.rolling(24, min_periods=24).max().reset_index(level=0, drop=True)
    df["price_roll_48_mean_shift96h"] = by_segment.rolling(48, min_periods=48).mean().reset_index(level=0, drop=True)
    df["price_roll_168_mean_shift96h"] = by_segment.rolling(168, min_periods=168).mean().reset_index(level=0, drop=True)
    return df


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["timestamp_local"] = df["timestamp"].dt.tz_convert(LOCAL_TZ)
    df["local_date"] = df["timestamp_local"].dt.date.astype(str)
    df["local_hour"] = df["timestamp_local"].dt.hour
    df["local_day_of_week"] = df["timestamp_local"].dt.dayofweek
    df["local_month"] = df["timestamp_local"].dt.month

    df["local_hour_sin"] = np.sin(2 * np.pi * df["local_hour"] / 24)
    df["local_hour_cos"] = np.cos(2 * np.pi * df["local_hour"] / 24)
    df["local_dow_sin"] = np.sin(2 * np.pi * df["local_day_of_week"] / 7)
    df["local_dow_cos"] = np.cos(2 * np.pi * df["local_day_of_week"] / 7)
    df["local_month_sin"] = np.sin(2 * np.pi * df["local_month"] / 12)
    df["local_month_cos"] = np.cos(2 * np.pi * df["local_month"] / 12)

    df["load_total_nl_de_mw"] = df["load_NL_load_forecast_mw"] + df["load_DE_LU_load_forecast_mw"]
    df["nl_load_share"] = df["load_NL_load_forecast_mw"] / df["load_total_nl_de_mw"].replace(0, np.nan)
    df["cross_border_net_flow_proxy_mw"] = df["flow_NL_NO_net_mw"] + df["flow_NL_GB_net_mw"]
    df["cross_border_abs_flow_mw"] = df["flow_NL_NO_net_mw"].abs() + df["flow_NL_GB_net_mw"].abs()
    df["wind_load_interaction"] = df["wind_speed_ms"] * df["load_NL_load_forecast_mw"]
    df["solar_cloud_interaction"] = df["solar_ghi"] * (100 - df["solar_cloud_cover"])
    df["workday_load_interaction"] = df["cal_is_working_day"].astype(float) * df["load_NL_load_forecast_mw"]

    df = add_gap_safe_target_features(df)
    required_history = [c for c in df.columns if c.startswith("price_lag_") or c.startswith("price_roll_")]
    return df.dropna(subset=required_history).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in EXCLUDED_MODEL_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            cols.append(col)
    return cols


def split_final_test(engineered: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_timestamps = set(raw.tail(FINAL_TEST_ROWS)["timestamp"])
    test = engineered[engineered["timestamp"].isin(final_timestamps)].copy()
    train = engineered[~engineered["timestamp"].isin(final_timestamps)].copy()
    if len(test) != FINAL_TEST_ROWS:
        raise ValueError(f"Expected {FINAL_TEST_ROWS} final test rows after engineering, found {len(test)}")
    return train.reset_index(drop=True), test.reset_index(drop=True)
