from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import TARGET
from .metrics import metric_dict
from .paths import FINAL_DIR, WF_DIR


def annotate_metrics(ax: plt.Axes, metrics: dict[str, float]) -> None:
    text = f"MAE {metrics['MAE']:.2f}\nRMSE {metrics['RMSE']:.2f}\nR2 {metrics['R2']:.3f}\nBias {metrics['bias']:.2f}"
    ax.text(0.01, 0.98, text, transform=ax.transAxes, va="top", ha="left", bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"})


def plot_final_results(preds: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    graphs = FINAL_DIR / "graphs"
    preds = preds.copy()
    preds["timestamp_local_plot"] = pd.to_datetime(preds["timestamp_local"])
    overall = metrics_df[(metrics_df["model"] == "FFNN") & (metrics_df["scope"] == "final_96h")].iloc[0].to_dict()

    plt.figure(figsize=(13, 6))
    ax = plt.gca()
    ax.plot(preds["timestamp_local_plot"], preds[TARGET], label="Actual", linewidth=2)
    ax.plot(preds["timestamp_local_plot"], preds["prediction_ffnn"], label="FFNN prediction", linewidth=2)
    for _, day_df in preds.groupby("local_date"):
        ax.axvline(day_df["timestamp_local_plot"].iloc[0], color="#999999", linewidth=0.8, alpha=0.5)
    ax.set_title("Final unseen 96 hours: predicted vs actual Dutch day-ahead prices")
    ax.set_xlabel("Local timestamp, Europe/Amsterdam")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    annotate_metrics(ax, overall)
    plt.tight_layout()
    plt.savefig(graphs / "primary_final_4day_predicted_vs_actual.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    axes = axes.ravel()
    for ax, (local_date, day_df) in zip(axes, preds.groupby("local_date")):
        day_metrics = metric_dict(day_df[TARGET].to_numpy(dtype=float), day_df["prediction_ffnn"].to_numpy(dtype=float))
        ax.plot(day_df["local_hour"], day_df[TARGET], marker="o", label="Actual")
        ax.plot(day_df["local_hour"], day_df["prediction_ffnn"], marker="o", label="Predicted")
        ax.set_title(local_date)
        ax.set_xlabel("Local hour")
        ax.set_ylabel("Price (EUR/MWh)")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(True, alpha=0.25)
        annotate_metrics(ax, day_metrics)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(graphs / "day_by_day_final_4day_panels.png", dpi=180)
    plt.close(fig)

    for local_date, day_df in preds.groupby("local_date"):
        day_metrics = metric_dict(day_df[TARGET].to_numpy(dtype=float), day_df["prediction_ffnn"].to_numpy(dtype=float))
        plt.figure(figsize=(9, 5))
        ax = plt.gca()
        ax.plot(day_df["local_hour"], day_df[TARGET], marker="o", label="Actual")
        ax.plot(day_df["local_hour"], day_df["prediction_ffnn"], marker="o", label="FFNN prediction")
        ax.set_title(f"Predicted vs actual by local hour: {local_date}")
        ax.set_xlabel("Local hour")
        ax.set_ylabel("Price (EUR/MWh)")
        ax.set_xticks(range(24))
        ax.grid(True, alpha=0.25)
        ax.legend()
        annotate_metrics(ax, day_metrics)
        plt.tight_layout()
        plt.savefig(graphs / f"day_{local_date}_predicted_vs_actual.png", dpi=180)
        plt.close()

    plt.figure(figsize=(13, 4))
    ax = plt.gca()
    ax.axhline(0, color="#333333", linewidth=1)
    ax.plot(preds["timestamp_local_plot"], preds["residual_ffnn"], marker="o")
    ax.set_title("Final test residuals over time")
    ax.set_xlabel("Local timestamp, Europe/Amsterdam")
    ax.set_ylabel("Prediction - actual (EUR/MWh)")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(graphs / "residuals_over_time.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.hist(preds["residual_ffnn"], bins=24, edgecolor="white")
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("Final test residual distribution")
    ax.set_xlabel("Prediction - actual (EUR/MWh)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(graphs / "residual_distribution.png", dpi=180)
    plt.close()

    hourly = preds.groupby("local_hour")["residual_ffnn"].mean().reset_index()
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.bar(hourly["local_hour"], hourly["residual_ffnn"])
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Mean residual by local hour")
    ax.set_xlabel("Local hour")
    ax.set_ylabel("Mean residual (EUR/MWh)")
    ax.set_xticks(range(24))
    plt.tight_layout()
    plt.savefig(graphs / "residual_by_local_hour.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.scatter(preds[TARGET], preds["prediction_ffnn"], alpha=0.8)
    lims = [min(preds[TARGET].min(), preds["prediction_ffnn"].min()), max(preds[TARGET].max(), preds["prediction_ffnn"].max())]
    ax.plot(lims, lims, color="#333333", linewidth=1, label="Ideal")
    ax.set_title("Predicted vs actual scatter")
    ax.set_xlabel("Actual (EUR/MWh)")
    ax.set_ylabel("Predicted (EUR/MWh)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    annotate_metrics(ax, overall)
    plt.tight_layout()
    plt.savefig(graphs / "scatter_predicted_vs_actual.png", dpi=180)
    plt.close()

    per_day = metrics_df[metrics_df["scope"].str.startswith("day_")].copy()
    per_day["local_date"] = per_day["scope"].str.replace("day_", "", regex=False)
    x = np.arange(len(per_day))
    width = 0.38
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.bar(x - width / 2, per_day["MAE"], width, label="MAE")
    ax.bar(x + width / 2, per_day["RMSE"], width, label="RMSE")
    ax.set_title("Final test error by local day")
    ax.set_xlabel("Local date")
    ax.set_ylabel("EUR/MWh")
    ax.set_xticks(x)
    ax.set_xticklabels(per_day["local_date"], rotation=20)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(graphs / "per_day_mae_rmse.png", dpi=180)
    plt.close()


def plot_walk_forward(metrics_df: pd.DataFrame, preds_df: pd.DataFrame, histories_df: pd.DataFrame) -> None:
    graphs = WF_DIR / "graphs"
    ff_metrics = metrics_df[metrics_df["model"] == "FFNN"].copy()
    for fold, fold_df in preds_df.groupby("fold"):
        fold_df = fold_df.copy()
        fold_df["timestamp_local_plot"] = pd.to_datetime(fold_df["timestamp_local"])
        m = ff_metrics[ff_metrics["fold"] == fold].iloc[0].to_dict()
        plt.figure(figsize=(12, 5))
        ax = plt.gca()
        ax.plot(fold_df["timestamp_local_plot"], fold_df[TARGET], label="Actual", linewidth=2)
        ax.plot(fold_df["timestamp_local_plot"], fold_df["prediction_ffnn"], label="FFNN prediction", linewidth=2)
        ax.set_title(f"Walk-forward validation fold {fold}")
        ax.set_xlabel("Local timestamp, Europe/Amsterdam")
        ax.set_ylabel("Price (EUR/MWh)")
        ax.legend()
        ax.grid(True, alpha=0.25)
        annotate_metrics(ax, m)
        plt.tight_layout()
        plt.savefig(graphs / f"fold_{fold}_predicted_vs_actual.png", dpi=180)
        plt.close()

    pivot = metrics_df.pivot_table(index="fold", columns="model", values="MAE", aggfunc="first")
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    if len(pivot) > 8:
        pivot.plot(marker="o", ax=ax)
        ax.set_xticks(pivot.index)
        ax.set_xticklabels([str(int(x)) for x in pivot.index], rotation=45, ha="right")
    else:
        pivot.plot(kind="bar", ax=ax)
    ax.set_title("Walk-forward validation MAE by fold and model")
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(graphs / "fold_metrics_mae_comparison.png", dpi=180)
    plt.close()

    fold_summary = ff_metrics.sort_values("fold").copy()
    fold_summary["validation_start_plot"] = (
        pd.to_datetime(fold_summary["validation_start_local"], utc=True).dt.tz_convert("Europe/Amsterdam").dt.tz_localize(None)
    )
    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    ax.plot(fold_summary["validation_start_plot"], fold_summary["MAE"], marker="o", label="FFNN MAE")
    ax.axhline(fold_summary["MAE"].mean(), color="#333333", linewidth=1, linestyle="--", label="Mean MAE")
    ax.fill_between(
        fold_summary["validation_start_plot"],
        fold_summary["MAE"].mean() - fold_summary["MAE"].std(ddof=1),
        fold_summary["MAE"].mean() + fold_summary["MAE"].std(ddof=1),
        color="#999999",
        alpha=0.15,
        label="+/- 1 std",
    )
    ax.set_title("Expanded walk-forward FFNN MAE over validation time")
    ax.set_xlabel("Validation window start, Europe/Amsterdam")
    ax.set_ylabel("MAE (EUR/MWh)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(graphs / "expanded_walk_forward_mae_over_time.png", dpi=180)
    plt.close()

    avg_history = histories_df.groupby("epoch", as_index=False)[["train_loss", "val_loss"]].mean()
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.plot(avg_history["epoch"], avg_history["train_loss"], label="Training loss")
    ax.plot(avg_history["epoch"], avg_history["val_loss"], label="Validation loss")
    ax.set_title("Average FFNN training and validation loss across folds")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Scaled target loss")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(graphs / "average_training_validation_loss.png", dpi=180)
    plt.close()
