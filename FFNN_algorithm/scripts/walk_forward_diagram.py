from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

from src.paths import WF_DIR

TRAIN_START = pd.Timestamp("2025-12-15")
TRAIN_COLOR = "#E8956D"
VAL_COLOR   = "#FBBF8C"
GAP_COLOR   = "#CCCCCC"
DATA_GAP_START = pd.Timestamp("2026-03-19")
DATA_GAP_END   = pd.Timestamp("2026-04-14")


def _days(ts: pd.Timestamp) -> float:
    return (ts.tz_localize(None) if ts.tzinfo else ts - TRAIN_START).total_seconds() / 86400


def _r2_color(r2: float) -> str:
    if r2 >= 0.85:
        return "#1a7a1a"
    if r2 >= 0.70:
        return "#4caf50"
    if r2 >= 0.50:
        return "#e67e22"
    return "#c0392b"


def main() -> None:
    wf = pd.read_csv(WF_DIR / "walk_forward_metrics.csv")
    ffnn = wf[wf["model"] == "FFNN"].sort_values("fold").copy()
    ffnn["val_start"] = pd.to_datetime(ffnn["validation_start_local"], utc=True).dt.tz_localize(None)
    ffnn["val_end"]   = pd.to_datetime(ffnn["validation_end_local"],   utc=True).dt.tz_localize(None)

    n = len(ffnn)
    total_days = _days(ffnn["val_end"].max()) + 10

    fig, ax = plt.subplots(figsize=(17, 10))

    bar_h = 0.62
    label_shown = False

    for _, row in ffnn.iterrows():
        fold = int(row["fold"])
        y = n + 1 - fold  # fold 1 at top

        vs_days = _days(row["val_start"])
        ve_days = _days(row["val_end"]) + 1  # inclusive end

        # Training bar
        ax.barh(y, vs_days, left=0, height=bar_h,
                color=TRAIN_COLOR, edgecolor="white", linewidth=0.3, zorder=2)

        # Validation bar
        ax.barh(y, ve_days - vs_days, left=vs_days, height=bar_h,
                color=VAL_COLOR, edgecolor="#D4883A", linewidth=0.8, zorder=2)

        # "Train" text (centered in training region, white)
        train_mid = vs_days / 2
        if fold == 1:
            ax.text(train_mid, y, "Train", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
        else:
            ax.text(train_mid, y, "Train", ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold")

        # "Val" text rotated on validate bar
        val_mid = vs_days + (ve_days - vs_days) / 2
        ax.text(val_mid, y, "Val", ha="center", va="center",
                fontsize=5, color="#7a3a00", fontweight="bold", rotation=90)

        # Metrics to the right
        r2   = row["R2"]
        rmse = row["RMSE"]
        mae  = row["MAE"]
        col  = _r2_color(r2)
        ax.text(ve_days + 1.5, y,
                f"R²={r2:+.2f}  RMSE={rmse:.1f}  MAE={mae:.1f}",
                va="center", fontsize=6.2, color=col)

    # Data gap shading
    gap_x0 = _days(DATA_GAP_START)
    gap_x1 = _days(DATA_GAP_END)
    ax.axvspan(gap_x0, gap_x1, color=GAP_COLOR, alpha=0.35, zorder=1, label="Data gap (March DST)")

    # Final test block annotation
    test_start_days = _days(pd.Timestamp("2026-05-25"))
    test_end_days   = _days(pd.Timestamp("2026-05-28")) + 1
    ax.axvspan(test_start_days, test_end_days, color="#9B59B6", alpha=0.25, zorder=1, label="Final test (96h)")

    # X-axis: month ticks
    months = pd.date_range("2025-12-01", "2026-06-15", freq="MS")
    month_days = [_days(m) for m in months]
    ax.set_xticks(month_days)
    ax.set_xticklabels([m.strftime("%b '%y") for m in months], rotation=30, ha="right", fontsize=9)
    ax.set_xlim(-2, total_days + 55)
    ax.set_xlabel("Date (Europe/Amsterdam)", fontsize=10)

    # Y-axis: fold labels
    ax.set_yticks(range(1, n + 2))
    ax.set_yticklabels([""] + [str(n + 1 - i) for i in range(1, n + 1)], fontsize=8)
    ax.set_ylim(0.4, n + 0.8)
    ax.set_ylabel("Fold", fontsize=10)

    ax.set_title(
        "Walk-forward validation — expanding window  |  23 folds × 4-day windows  |  Dec 2025 – May 2026",
        fontsize=11, pad=12,
    )

    # Legend patches
    legend_handles = [
        mpatches.Patch(color=TRAIN_COLOR, label="Training data"),
        mpatches.Patch(color=VAL_COLOR,   label="Validation window (4 days)"),
        mpatches.Patch(color=GAP_COLOR,   alpha=0.5, label="Data gap (March DST)"),
        mpatches.Patch(color="#9B59B6",   alpha=0.4, label="Final held-out test"),
    ]
    # R² color legend
    for label, col in [("R² ≥ 0.85", "#1a7a1a"), ("R² 0.70–0.85", "#4caf50"),
                        ("R² 0.50–0.70", "#e67e22"), ("R² < 0.50", "#c0392b")]:
        legend_handles.append(mpatches.Patch(color=col, label=label))
    ax.legend(handles=legend_handles, loc="lower left", fontsize=7.5,
              ncol=2, framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)

    plt.tight_layout()
    out = PROJECT_ROOT / "results" / "walk_forward_diagram.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
