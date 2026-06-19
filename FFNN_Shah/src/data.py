from __future__ import annotations

import shutil

import pandas as pd

from .config import TARGET
from .paths import ORIGINAL_CSV, SOURCE_CSV


def copy_original() -> None:
    if ORIGINAL_CSV.exists():
        return
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Source CSV not found: {SOURCE_CSV}")
    shutil.copy2(SOURCE_CSV, ORIGINAL_CSV)


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(ORIGINAL_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def make_data_profile(raw: pd.DataFrame) -> dict[str, object]:
    diffs = raw["timestamp"].diff()
    gaps = raw.loc[diffs > pd.Timedelta(hours=1), ["timestamp"]].copy()
    gap_rows = []
    for idx in gaps.index:
        previous_ts = raw.loc[idx - 1, "timestamp"]
        current_ts = raw.loc[idx, "timestamp"]
        gap_rows.append(
            {
                "row": int(idx),
                "previous_timestamp_utc": str(previous_ts),
                "current_timestamp_utc": str(current_ts),
                "gap": str(current_ts - previous_ts),
            }
        )

    feb_added = raw[
        (raw["timestamp"] >= pd.Timestamp("2026-02-05 23:00:00+00:00"))
        & (raw["timestamp"] <= pd.Timestamp("2026-02-28 22:00:00+00:00"))
    ]
    return {
        "shape": raw.shape,
        "start": str(raw["timestamp"].min()),
        "end": str(raw["timestamp"].max()),
        "duplicate_timestamps": int(raw["timestamp"].duplicated().sum()),
        "missing_values": int(raw.isna().sum().sum()),
        "target_min": float(raw[TARGET].min()),
        "target_max": float(raw[TARGET].max()),
        "target_mean": float(raw[TARGET].mean()),
        "negative_target_prices": int((raw[TARGET] < 0).sum()),
        "gaps": gap_rows,
        "feb_rows_added_period_count": int(len(feb_added)),
    }
