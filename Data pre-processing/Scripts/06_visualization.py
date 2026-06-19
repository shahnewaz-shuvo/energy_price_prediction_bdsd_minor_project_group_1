# Script 06 - plots / validation
# makes the graphs from the clean csv. they double as a check that the cleaned


import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# paths
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_regimeB")
INPUT_CSV  = os.path.join(OUTPUT_DIR, "regime_B_clean.csv")
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# style
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3, "axes.edgecolor": "#888888",
})
C_MAIN = "#2c7fb8"
C_LINE = "#253494"

# the gas crisis window, just for context on the spring spikes
CRISIS_START = pd.Timestamp("2026-02-28")
CRISIS_END   = pd.Timestamp("2026-04-08")

print("=" * 65)
print("SCRIPT 06 - PLOTS")
print("=" * 65)
print(f"Input : {INPUT_CSV}")
print(f"Output: {FIG_DIR}")
print()

if not os.path.exists(INPUT_CSV):
    print("ERROR: regime_B_clean.csv not found. Run Script 04 first.")
    raise SystemExit

# load
print("STEP 1 - Loading clean dataset")
print("-" * 50)
df = pd.read_csv(INPUT_CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)
df["t"] = df["timestamp"].dt.tz_convert(None)
print(f"  Loaded: {len(df):,} rows x {df.shape[1]} columns")
print(f"  Date range: {df['t'].min()} -> {df['t'].max()}")
print()

# figure 0 - raw price (read from the parsed file, before cleaning)
print("STEP 2 - Figure 0: raw parsed price")
print("-" * 50)
RAW_ENERGY = os.path.join(OUTPUT_DIR, "parsed_energy.parquet")
if os.path.exists(RAW_ENERGY):
    raw = pd.read_parquet(RAW_ENERGY)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.sort_values("timestamp")
    raw["t"] = raw["timestamp"].dt.tz_convert(None)
    print(f"  raw rows                     : {len(raw):,}")
    print(f"  duplicate timestamps         : {raw['timestamp'].duplicated().sum():,}")
    print(f"  missing ENTSO-E price values : {raw['price_entsoe'].isna().sum():,}")
    rp = raw.dropna(subset=["price_entsoe"])
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(rp["t"], rp["price_entsoe"], lw=0.5, color="#888888")
    ax.set_title("Raw ENTSO-E price straight from the JSON files "
                 "(before merging and cleaning)", fontweight="bold")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "00_raw_price.png"))
    plt.close()
    print("  Saved 00_raw_price.png\n")
else:
    print("  parsed_energy.parquet not found - skipping raw plot\n")

# figure 1 - full clean price series, with the crisis band and the min/max marked
print("STEP 3 - Figure 1: full price series")
print("-" * 50)
imin, imax = df["price_entsoe"].idxmin(), df["price_entsoe"].idxmax()
print(f"  Min: {df.loc[imin,'price_entsoe']:.1f} on {df.loc[imin,'t']:%d %b %Y %H:%M}")
print(f"  Max: {df.loc[imax,'price_entsoe']:.1f} on {df.loc[imax,'t']:%d %b %Y %H:%M}")

fig, ax = plt.subplots(figsize=(11, 4.0))
ax.set_ylim(-580, 470)   # room so the spike labels stay inside the plot
ax.axvspan(CRISIS_START, CRISIS_END, color="#fdae6b", alpha=0.25,
           label="Strait of Hormuz gas crisis (Feb-Apr 2026)")
ax.plot(df["t"], df["price_entsoe"], lw=0.6, color=C_MAIN)
ax.axhline(0, color="red", lw=0.8, alpha=0.6)
ax.annotate(f"min EUR {df.loc[imin,'price_entsoe']:.0f} (1 May)",
            (df.loc[imin, "t"], df.loc[imin, "price_entsoe"]),
            xytext=(50, 30), textcoords="offset points", ha="left", va="center", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="red"))
ax.annotate(f"max EUR {df.loc[imax,'price_entsoe']:.0f}",
            (df.loc[imax, "t"], df.loc[imax, "price_entsoe"]),
            xytext=(-14, -16), textcoords="offset points", ha="right", va="top", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="k"))
ax.set_title("ENTSO-E Day-Ahead Electricity Price (NL) - Dec 2025 to May 2026",
             fontweight="bold")
ax.set_ylabel("Price (EUR/MWh)")
ax.legend(loc="upper left", fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_price_timeseries.png"))
plt.close()
print("  Saved 01_price_timeseries.png\n")

# figure 2 - average price by hour (the daily cycle)
print("STEP 4 - Figure 2: daily cycle")
print("-" * 50)
hourly = df.groupby("cal_hour")["price_entsoe"].mean()
hstd   = df.groupby("cal_hour")["price_entsoe"].std()
print(f"  04:00 {hourly[4]:.1f} | 08:00 {hourly[8]:.1f} | 13:00 {hourly[13]:.1f} | 17:00 {hourly[17]:.1f}")
fig, ax = plt.subplots(figsize=(7, 4))
morn_h = int(hourly.loc[5:11].idxmax())   # the real morning peak hour
ax.plot(hourly.index, hourly.values, marker="o", color=C_LINE)
ax.fill_between(hourly.index, hourly - hstd, hourly + hstd, alpha=0.15, color=C_LINE)
ax.set_ylim(-60, 210)
ax.set_title("Average Price by Hour of Day", fontweight="bold")
ax.set_xlabel("Hour of day (UTC)")
ax.set_ylabel("Price (EUR/MWh)")
ax.set_xticks(range(0, 24, 2))
ax.annotate("morning peak", (morn_h, hourly[morn_h]), xytext=(morn_h, hourly[morn_h] + 42),
            ha="center", fontsize=8, arrowprops=dict(arrowstyle="->"))
ax.annotate("solar midday dip", (13, hourly[13]), xytext=(13, -46),
            ha="center", fontsize=8, arrowprops=dict(arrowstyle="->"))
ax.annotate("evening peak", (int(hourly.idxmax()), hourly.max()),
            xytext=(int(hourly.idxmax()), hourly.max() + 42),
            ha="center", fontsize=8, arrowprops=dict(arrowstyle="->"))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_daily_cycle.png"))
plt.close()
print("  Saved 02_daily_cycle.png\n")

# figure 3 - weekday/weekend and price by month
print("STEP 5 - Figure 3: weekday/weekend + monthly")
print("-" * 50)
is_we = df["cal_day_of_week"].isin([5, 6])
print(f"  weekday mean {df[~is_we]['price_entsoe'].mean():.1f} | weekend mean {df[is_we]['price_entsoe'].mean():.1f}")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
data = [df[~is_we]["price_entsoe"], df[is_we]["price_entsoe"]]
bp = axes[0].boxplot(data, tick_labels=["Weekday", "Weekend"], patch_artist=True, showfliers=False)
for patch in bp["boxes"]:
    patch.set_facecolor(C_MAIN)
axes[0].set_title("Weekday vs Weekend", fontweight="bold")
axes[0].set_ylabel("Price (EUR/MWh)")
order = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
mdata = [df[df["source_month"] == m]["price_entsoe"] for m in order]
bp2 = axes[1].boxplot(mdata, tick_labels=[m[2:] for m in order], patch_artist=True, showfliers=False)
for patch in bp2["boxes"]:
    patch.set_facecolor(C_MAIN)
axes[1].set_title("Price Distribution by Month", fontweight="bold")
axes[1].set_ylabel("Price (EUR/MWh)")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_weekday_monthly.png"))
plt.close()
print("  Saved 03_weekday_monthly.png\n")

# figure 4 - overall price distribution
print("STEP 6 - Figure 4: price distribution")
print("-" * 50)
p = df["price_entsoe"]
print(f"  mean {p.mean():.1f} | std {p.std():.1f} | negatives {int((p<0).sum())}")
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(p, bins=70, color=C_MAIN, alpha=0.8)
ax.axvline(0, color="red", lw=1, label="zero price")
ax.axvline(p.mean(), color="k", lw=1, ls="--", label=f"mean EUR {p.mean():.0f}")
ax.set_title("Distribution of ENTSO-E Prices", fontweight="bold")
ax.set_xlabel("Price (EUR/MWh)")
ax.set_ylabel("Number of hours")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_price_distribution.png"))
plt.close()
print("  Saved 04_price_distribution.png\n")

# figure 5 - data completeness per month
print("STEP 7 - Figure 5: completeness")
print("-" * 50)
full = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
present = set(df["timestamp"])
present_mask = pd.Series([ts in present for ts in full])
month_lbl = pd.Series(full).dt.strftime("%Y-%m")
tab = pd.DataFrame({"month": month_lbl, "present": present_mask.values})
g = tab.groupby("month")["present"].agg(["sum", "count"])
g["missing"] = g["count"] - g["sum"]
# only the 6 months we cover (the range starts at 23:00 on 30 Nov, ignore that stray bar)
keep_months = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
g = g.reindex(keep_months)
print(f"  expected slots {len(full):,} | present {len(df):,} | missing {len(full)-len(df):,}")
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_ylim(0, g["count"].max() * 1.20)
ax.bar(g.index, g["sum"], color=C_MAIN, label="Present hours")
ax.bar(g.index, g["missing"], bottom=g["sum"], color="#e34a33", label="Missing hours")
for i, (m, r) in enumerate(g.iterrows()):
    if r["missing"] > 0:
        ax.text(i, r["sum"] + r["missing"] / 2, f"{int(r['missing'])}h\nmissing",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")
ax.set_title("Hourly Data Completeness by Month", fontweight="bold")
ax.set_ylabel("Hours")
ax.tick_params(axis="x", rotation=45)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "06_completeness.png"))
plt.close()
print("  Saved 06_completeness.png\n")

# figure 6 - feature correlation (shown last in the report)
print("STEP 8 - Figure 6: correlations")
print("-" * 50)
feats = ["solar_ghi", "solar_cloud_cover", "wind_speed_ms", "dw_wind_speed",
         "dw_temperature", "dw_humidity", "dw_hdd", "load_NL_load_forecast_mw",
         "load_DE_LU_load_forecast_mw", "flow_NL_NO_net_mw", "flow_NL_GB_net_mw",
         "gen_FR_nuclear_actual_mw", "grid_balance_delta_mw",
         "cal_is_working_day", "cal_is_holiday_nl"]
feats = [f for f in feats if f in df.columns]
cors = df[feats].corrwith(df["price_entsoe"]).sort_values()
for name, v in cors.items():
    print(f"  {name:<30} {v:+.3f}")
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(range(len(cors)), cors.values,
        color=["#d73027" if v < 0 else "#1a9850" for v in cors.values])
ax.set_yticks(range(len(cors)))
ax.set_yticklabels(cors.index, fontsize=9)
ax.axvline(0, color="k", lw=0.8)
ax.set_title("Feature Correlation with ENTSO-E Price", fontweight="bold")
ax.set_xlabel("Pearson r")
for i, v in enumerate(cors.values):
    ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:+.2f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_correlations.png"))
plt.close()
print("  Saved 05_correlations.png\n")

# clean up the old two-regime figure if it's still around
old = os.path.join(FIG_DIR, "04_measured_vs_predicted.png")
if os.path.exists(old):
    os.remove(old)
    print("  removed old 04_measured_vs_predicted.png")

print("=" * 65)
print("DONE - figures saved to output_regimeB/figures/")
print("=" * 65)
