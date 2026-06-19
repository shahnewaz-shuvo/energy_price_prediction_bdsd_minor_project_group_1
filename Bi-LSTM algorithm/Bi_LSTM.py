"""
Bi-LSTM 
"""

# %% Section 1: Imports
import copy
import os
import pandas as pd
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# %% Section 2: Configuration
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regime_B_features_selected.csv")

SEQ_LEN    = 168    # Input look-back window (168 h = 7 days)
FORECAST_H = 72     # Forecast horizon (72 h = 3 days) — fixed task requirement

# Hyperparameter search space
HIDDEN_SIZE_VALUES = [64, 128, 256]                 # Neurons in hidden layer
DROPOUT_VALUES     = [0.1, 0.2, 0.3]                # Dropout rate | regularisation
BATCH_SIZE_VALUES  = [64, 128, 256]                 # Batch size
PRED_STEPS_VALUES  = [12, 24, 72]                   # Model output chunk: 72 = direct multi-output,
                                                    # <72 = applied recursively to cover 72 h

NUM_LAYERS = 2                                      # Stacked LSTM, fixed to shrink the search space
LR         = 1e-3                                   # Fixed parameter, is decreased by scheduler during training (on plateau)
EPOCHS     = 50                                     # Max epochs per fold (early stopping likely to kick in well before this)
PATIENCE   = 5                                      # max number of epochs with no val loss improvement

WF_N_FOLDS  = 20                                    # Number of rolling walk-forward folds
WF_VAL_DAYS = 3                                     # Validation window per fold (days)
WF_VAL_H    = WF_VAL_DAYS * 24                      # Validation window per fold (hours)

TUNE_FOLD_STRIDE = 4                                # Tune on every 4th fold; final CV uses all folds

TEST_H = 96                                         # Test set = last 96 hours 

SEED = 42

TARGET_COL     = "price_entsoe"
LOCAL_TIMEZONE = "Europe/Amsterdam"

data_features = [
    "load_NL_load_forecast_mw",
    "flow_NL_GB_net_mw",
    "flow_NL_NO_net_mw",
    "load_DE_LU_load_forecast_mw",
    "dw_humidity",
    "dw_temperature",
    "dw_wind_speed",
    "solar_ghi",   
]

time_features = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
base_features = data_features + time_features

PAST_SIZE   = len(base_features) + 1   
FUTURE_SIZE = len(base_features)

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% Section 3: Model
class BiLSTM(nn.Module):

    def __init__(self, past_input_size, future_input_size, hidden_size,
                 num_layers, pred_steps, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=past_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.direction_score = nn.Linear(hidden_size * 2, 2)

        combined_size = hidden_size + pred_steps * future_input_size
        self.fc = nn.Sequential(
            nn.LayerNorm(combined_size),
            nn.Linear(combined_size, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),           nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, pred_steps),
        )

    def forward(self, x_past, x_future):
        _, (h_n, _) = self.lstm(x_past)
        h_fwd, h_bwd = h_n[-2], h_n[-1]

        weights = torch.softmax(
            self.direction_score(torch.cat([h_fwd, h_bwd], dim=1)), dim=1
        )
        lstm_features   = weights[:, 0:1] * h_fwd + weights[:, 1:2] * h_bwd
        future_features = x_future.reshape(x_future.size(0), -1)
        return self.fc(torch.cat([lstm_features, future_features], dim=1))


# %% Section 4: Helper functions
def fit_standardizer(values):
    mu    = np.nanmean(values, axis=0)
    sigma = np.nanstd(values, axis=0)
    mu    = np.where(np.isnan(mu), 0.0, mu)
    sigma = np.where((sigma == 0) | np.isnan(sigma), 1.0, sigma)
    return mu, sigma


def create_sequences(X_past, X_future, y,
                     win_start, win_end,
                     seq_len, pred_steps,
                     context_lookback=True):

    xs_past, xs_future, ys = [], [], []

    i_first = max(0, win_start - seq_len) if context_lookback else win_start
    i_last  = win_end - seq_len - pred_steps

    for i in range(i_first, i_last + 1):
        target_start = i + seq_len
        target_end   = target_start + pred_steps
        if target_start >= win_start and target_end <= win_end:
            xs_past.append(X_past[i : i + seq_len])
            xs_future.append(X_future[target_start : target_end])
            ys.append(y[target_start : target_end])

    return (np.asarray(xs_past,   dtype=np.float32),
            np.asarray(xs_future, dtype=np.float32),
            np.asarray(ys,        dtype=np.float32))


def make_loader(Xp, Xf, y, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(Xp).float(),
        torch.from_numpy(Xf).float(),
        torch.from_numpy(y).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def build_model(hidden_size, num_layers, dropout, pred_steps):
    return BiLSTM(
        past_input_size=PAST_SIZE,
        future_input_size=FUTURE_SIZE,
        hidden_size=hidden_size,
        num_layers=num_layers,
        pred_steps=pred_steps,
        dropout=dropout,
    ).to(device)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for Xp, Xf, yb in loader:
        Xp, Xf, yb = Xp.to(device), Xf.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xp, Xf), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        n          += len(yb)
    return total_loss / n


def forecast_horizon(model, Xp, Xf, origins, pred_steps, device):

    assert FORECAST_H % pred_steps == 0, "pred_steps must divide FORECAST_H"
    model.eval()
    xp_win = np.stack([Xp[t - SEQ_LEN : t] for t in origins]).astype(np.float32)
    preds  = []
    with torch.no_grad():
        for s in range(0, FORECAST_H, pred_steps):
            xf_step = np.stack(
                [Xf[t + s : t + s + pred_steps] for t in origins]
            ).astype(np.float32)
            p = model(
                torch.from_numpy(xp_win).to(device),
                torch.from_numpy(xf_step).to(device),
            ).cpu().numpy()
            preds.append(p)
            # roll the window forward: known future covariates + predicted price
            new_rows = np.concatenate([xf_step, p[..., None]], axis=2)
            xp_win   = np.concatenate([xp_win[:, pred_steps:], new_rows], axis=1)
    return np.concatenate(preds, axis=1)


def make_val_fn(fd, pred_steps):
    t0     = fd["val_start"]
    y_true = torch.from_numpy(fd["y"][t0 : t0 + FORECAST_H]).float().unsqueeze(0)

    def val_fn(model):
        preds = forecast_horizon(model, fd["Xp"], fd["Xf"], [t0], pred_steps, device)
        return criterion(torch.from_numpy(preds), y_true).item()

    return val_fn


def regression_metrics(preds, actuals):
    errors = preds - actuals
    mae    = np.mean(np.abs(errors))
    rmse   = np.sqrt(np.mean(errors ** 2))
    bias   = np.mean(errors)
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return mae, rmse, r2, bias


def train_model(model, train_loader, val_fn, criterion):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5
    )
    best_val, best_state, no_improve = float("inf"), None, 0
    history = {"train": [], "val": []}

    for _ in range(1, EPOCHS + 1):
        tl = train_epoch(model, train_loader, optimizer, criterion, device)
        vl = val_fn(model)
        history["train"].append(tl)
        history["val"].append(vl)
        scheduler.step(vl)

        if vl < best_val - 1e-5:
            best_val   = vl
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return best_val, history


# %% Section 5: Load data and create features
data = pd.read_csv(CSV_PATH)
data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
data = data.sort_values("timestamp").reset_index(drop=True)

timestamp_local   = data["timestamp"].dt.tz_convert(LOCAL_TIMEZONE)
data["hour"]      = timestamp_local.dt.hour
data["dayofweek"] = timestamp_local.dt.dayofweek
data["hour_sin"]  = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"]  = np.cos(2 * np.pi * data["hour"] / 24)
data["day_sin"]   = np.sin(2 * np.pi * data["dayofweek"] / 7)
data["day_cos"]   = np.cos(2 * np.pi * data["dayofweek"] / 7)

base_raw  = data[base_features].to_numpy(dtype=np.float32)
price_raw = data[TARGET_COL].to_numpy(dtype=np.float32)
total_hours = len(data)


# %% Section 6: Compute walk-forward split parameters (expanding window)
cv_hours   = total_hours - TEST_H
WF_TRAIN_H = cv_hours - WF_N_FOLDS * WF_VAL_H   # initial training window

print("=" * 60)
print("Walk-forward split parameters (expanding window)")
print("=" * 60)
print(f"  Total hours:         {total_hours}  ({total_hours/24:.0f} days)")
print(f"  CV hours:            {cv_hours}  ({cv_hours/24:.0f} days)")
print(f"  Test set:            last {TEST_H} h (held out)")
print(f"  Folds:               {WF_N_FOLDS}")
print(f"  Initial train window:{WF_TRAIN_H} h  ({WF_TRAIN_H/24:.0f} days)")
print(f"  Val window/fold:     {WF_VAL_H} h  ({WF_VAL_DAYS} days)")
print()
for k in range(WF_N_FOLDS):
    te = WF_TRAIN_H + k * WF_VAL_H     # train end expands each fold
    print(f"  Fold {k+1:>2}: train [0, {te:>6})  val [{te:>6}, {te+WF_VAL_H:>6})")
print(f"  Test:       [{cv_hours}, {total_hours})")
print()


# %% Section 7: Bayesian optimisation — mean validation loss over every 4th fold
optuna.logging.set_verbosity(optuna.logging.WARNING)

criterion = nn.HuberLoss()

fold_datasets = []
for fold in range(WF_N_FOLDS):
    train_start = 0                                     # always anchored at 0
    train_end   = WF_TRAIN_H + fold * WF_VAL_H          # grows each fold
    val_start   = train_end
    val_end     = val_start + WF_VAL_H

    # Normalise on this fold's training window only (no leakage)
    mu_b, sig_b = fit_standardizer(base_raw[train_start:train_end])
    mu_y, sig_y = fit_standardizer(price_raw[train_start:train_end])
    mu_y, sig_y = float(mu_y), float(sig_y)

    b_norm = ((base_raw  - mu_b) / sig_b).astype(np.float32)
    p_norm = ((price_raw - mu_y) / sig_y).astype(np.float32)

    Xp = np.concatenate([b_norm, p_norm.reshape(-1, 1)], axis=1)
    Xf = b_norm
    y  = p_norm

    fold_datasets.append(dict(
        fold=fold,
        train_start=train_start, train_end=train_end,
        val_start=val_start,     val_end=val_end,
        mu_y=mu_y, sig_y=sig_y,
        Xp=Xp, Xf=Xf, y=y,
    ))

# Tuning subset: every TUNE_FOLD_STRIDE-th fold, ending at the last fold (which
# has the most training data). The final CV in Section 9 still uses all folds.
tuning_folds = fold_datasets[TUNE_FOLD_STRIDE - 1 :: TUNE_FOLD_STRIDE]
print(f"Tuning on folds {[fd['fold'] + 1 for fd in tuning_folds]} "
      f"({len(tuning_folds)} of {WF_N_FOLDS})")


# ── Objective function ────────────────────────────────────────────────────────
def objective(trial):
    hidden_size = trial.suggest_categorical("hidden_size", HIDDEN_SIZE_VALUES)
    dropout     = trial.suggest_categorical("dropout",     DROPOUT_VALUES)
    batch_size  = trial.suggest_categorical("batch_size",  BATCH_SIZE_VALUES)
    pred_steps  = trial.suggest_categorical("pred_steps",  PRED_STEPS_VALUES)

    fold_losses = []
    for i, fd in enumerate(tuning_folds):
        Xp_tr, Xf_tr, y_tr = create_sequences(
            fd["Xp"], fd["Xf"], fd["y"], fd["train_start"], fd["train_end"],
            SEQ_LEN, pred_steps, context_lookback=False)

        model       = build_model(hidden_size, NUM_LAYERS, dropout, pred_steps)
        best_val, _ = train_model(
            model,
            make_loader(Xp_tr, Xf_tr, y_tr, batch_size, shuffle=True),
            make_val_fn(fd, pred_steps),
            criterion,
        )
        fold_losses.append(best_val)

        # Report the running mean so the pruner can stop unpromising trials early
        trial.report(float(np.mean(fold_losses)), step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_losses))


# ── Run the study ─────────────────────────────────────────────────────────────
N_TRIALS = 30   # search space is 3×3×3×3 = 81 combos; TPE + Hyperband pruning
                # explores it far cheaper than the exhaustive grid

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=len(tuning_folds)),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ── Results ───────────────────────────────────────────────────────────────────
best_params = study.best_params
best_gs_val = study.best_value

print(f"\nBest hyperparameters:  {best_params}")
print(f"Best mean CV val loss: {best_gs_val:.4f}")

grid_results = [
    {
        "hidden_size": t.params["hidden_size"],
        "dropout":     t.params["dropout"],
        "batch_size":  t.params["batch_size"],
        "pred_steps":  t.params["pred_steps"],
        "val_loss":    t.value,
    }
    for t in study.trials
    if t.state == optuna.trial.TrialState.COMPLETE
]
results_gs_df = pd.DataFrame(grid_results).sort_values("val_loss").reset_index(drop=True)
print("\nOptuna trial results (sorted by mean CV val loss):")
print(results_gs_df.to_string(index=False))

print("\nBest mean CV val loss per forecast strategy:")
for ps, grp in results_gs_df.groupby("pred_steps"):
    label = "direct multi-output" if ps == FORECAST_H else f"recursive, {ps} h/step"
    print(f"  pred_steps={ps:>3} ({label}): {grp['val_loss'].min():.4f}")

HIDDEN_SIZE = best_params["hidden_size"]
DROPOUT     = best_params["dropout"]
BATCH_SIZE  = best_params["batch_size"]
PRED_STEPS  = best_params["pred_steps"]


# %% Section 8: Optuna study diagnostics — history, importance, sampling

completed_trials = [t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE]
trial_nums   = np.array([t.number for t in completed_trials])
trial_losses = np.array([t.value  for t in completed_trials])

# ── Figure 1: optimisation history ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(trial_nums, trial_losses, s=35, color="tab:blue", alpha=0.7,
           label="Trial")
ax.plot(trial_nums, np.minimum.accumulate(trial_losses), color="tab:orange",
        linewidth=2, drawstyle="steps-post", label="Best so far")
ax.set_xlabel("Trial number")
ax.set_ylabel("Mean CV val loss (Huber)")
ax.set_title("Optuna Optimisation History (completed trials)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# ── Figure 2: parameter importances ────────────────────────────────────
try:
    importances = optuna.importance.get_param_importances(study)
    imp_names  = list(importances.keys())      # ordered most → least important
    imp_scores = list(importances.values())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(imp_names, imp_scores, color="tab:blue", alpha=0.8)
    ax.invert_yaxis()                           # most important bar on top
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_xlim(0, max(imp_scores) * 1.15)
    ax.set_title("Hyperparameter Importance")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Warning: skipping parameter-importance plot "
          f"(importance computation failed: {e})")

# ── Figure 3: sampled value per trial, per hyperparameter ─────────────────────
param_value_grid = {
    "hidden_size": HIDDEN_SIZE_VALUES,
    "dropout":     DROPOUT_VALUES,
    "batch_size":  BATCH_SIZE_VALUES,
    "pred_steps":  PRED_STEPS_VALUES,
}

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
loss_norm = plt.Normalize(trial_losses.min(), trial_losses.max())

for ax, (param, values) in zip(axes.ravel(), param_value_grid.items()):
    sc = ax.scatter(trial_nums,
                    [t.params[param] for t in completed_trials],
                    c=trial_losses, cmap="RdYlGn_r", norm=loss_norm,
                    s=45, edgecolors="grey", linewidths=0.5)
    ax.set_yticks(values)
    ax.set_title(param)
    ax.grid(True, alpha=0.3)

for ax in axes[-1]:
    ax.set_xlabel("Trial number")

fig.suptitle("Sampled Value per Trial, Coloured by Loss (completed trials)")
plt.tight_layout()
fig.colorbar(sc, ax=axes.ravel().tolist(),
             label="Mean CV val loss (Huber)", shrink=0.95)
plt.show()


# %% Section 9: Walk-forward CV — train all folds with best hyperparameters
# The last fold's trained model and normaliser are saved for test evaluation.
fold_results   = []
fold_histories = []

for fd in fold_datasets:
    fold = fd["fold"]

    Xp_tr, Xf_tr, y_tr = create_sequences(
        fd["Xp"], fd["Xf"], fd["y"], fd["train_start"], fd["train_end"],
        SEQ_LEN, PRED_STEPS, context_lookback=False)

    print(f"\nFold {fold+1}/{WF_N_FOLDS}  "
          f"train=[{fd['train_start']},{fd['train_end']})  "
          f"val=[{fd['val_start']},{fd['val_end']})  "
          f"train_seqs={len(Xp_tr):,}")

    model    = build_model(HIDDEN_SIZE, NUM_LAYERS, DROPOUT, PRED_STEPS)
    best_val, history = train_model(
        model,
        make_loader(Xp_tr, Xf_tr, y_tr, BATCH_SIZE, shuffle=True),
        make_val_fn(fd, PRED_STEPS),
        criterion,
    )
    fold_histories.append((fold + 1, history))

    # Fold metrics on the full 72 h forecast from the validation origin
    val_preds_n   = forecast_horizon(
        model, fd["Xp"], fd["Xf"], [fd["val_start"]], PRED_STEPS, device)
    val_actuals_n = fd["y"][fd["val_start"] : fd["val_start"] + FORECAST_H][None, :]

    val_preds   = val_preds_n   * fd["sig_y"] + fd["mu_y"]
    val_actuals = val_actuals_n * fd["sig_y"] + fd["mu_y"]

    mae, rmse, r2, bias = regression_metrics(val_preds, val_actuals)
    fold_results.append(dict(fold=fold + 1, mae=mae, rmse=rmse, r2=r2,
                             bias=bias, val_loss=best_val))

    print(f"  → MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}  bias={bias:.2f}")

    # Save last fold artefacts for test evaluation
    if fold == WF_N_FOLDS - 1:
        last_fold_model = model
        last_fold_mu_y  = fd["mu_y"]
        last_fold_sig_y = fd["sig_y"]
        last_fold_Xp    = fd["Xp"]
        last_fold_Xf    = fd["Xf"]
        last_fold_y     = fd["y"]


# %% Section 10: Mean training vs validation loss per epoch (across folds)
# Folds early-stop at different epochs, so each epoch's mean is taken over the
# folds still training at that epoch (NaN-padded). The grey line shows how
# many folds contribute — read the tail of the curves with that in mind.
max_epochs = max(len(h["train"]) for _, h in fold_histories)
train_mat  = np.full((len(fold_histories), max_epochs), np.nan)
val_mat    = np.full((len(fold_histories), max_epochs), np.nan)
for r, (_, h) in enumerate(fold_histories):
    train_mat[r, :len(h["train"])] = h["train"]
    val_mat[r,   :len(h["val"])]   = h["val"]

epochs       = np.arange(1, max_epochs + 1)
mean_train   = np.nanmean(train_mat, axis=0)
mean_val     = np.nanmean(val_mat, axis=0)
active_folds = np.sum(~np.isnan(val_mat), axis=0)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(epochs, mean_train, label="Mean train loss",
        linewidth=2, color="tab:blue")
ax.plot(epochs, mean_val,   label="Mean validation loss",
        linewidth=2, color="tab:orange")
ax.set_xlabel("Epoch")
ax.set_ylabel("Huber loss (normalised)")
ax.grid(True, alpha=0.3)

ax2 = ax.twinx()
ax2.plot(epochs, active_folds, color="grey", linestyle=":", linewidth=1.5,
         label="Folds still training")
ax2.set_ylabel("Folds still training", color="grey")
ax2.set_ylim(0, len(fold_histories) + 1)
ax2.tick_params(axis="y", colors="grey")

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

ax.set_title("Mean Training vs Validation Loss per Epoch (across folds)")
plt.tight_layout()
plt.show()


# %% Section 11: CV results summary
results_df = pd.DataFrame(fold_results)

print("\n" + "=" * 60)
print("Walk-Forward CV Summary")
print("=" * 60)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\nMean MAE:  {results_df['mae'].mean():.2f} EUR/MWh")
print(f"Mean RMSE: {results_df['rmse'].mean():.2f} EUR/MWh")
print(f"Mean R²:   {results_df['r2'].mean():.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, label, color in zip(
    axes,
    ["mae", "rmse", "r2"],
    ["MAE EUR/MWh", "RMSE EUR/MWh", "R²"],
    ["tab:blue", "tab:orange", "tab:green"],
):
    mean_val = results_df[col].mean()
    ax.bar(results_df["fold"], results_df[col], color=color, alpha=0.8)
    ax.axhline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.2f}")
    ax.set_title(f"{label} per Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel(label)
    ax.set_xticks(results_df["fold"])
    ax.legend()

fig.suptitle("Expanding Walk-Forward Cross-Validation Results")
plt.tight_layout()
plt.show()


# %% Section 12: Test evaluation

test_origins = np.arange(cv_hours, total_hours - FORECAST_H + 1)

test_preds_n   = forecast_horizon(
    last_fold_model, last_fold_Xp, last_fold_Xf, test_origins, PRED_STEPS, device)
test_actuals_n = np.stack([last_fold_y[t : t + FORECAST_H] for t in test_origins])

test_preds   = test_preds_n   * last_fold_sig_y + last_fold_mu_y
test_actuals = test_actuals_n * last_fold_sig_y + last_fold_mu_y

test_mae, test_rmse, test_r2, test_bias = regression_metrics(test_preds, test_actuals)

print(f"\nTest set metrics (last {TEST_H} h, {len(test_origins)} forecast origins):")
print(f"  MAE:  {test_mae:.2f} EUR/MWh")
print(f"  RMSE: {test_rmse:.2f} EUR/MWh")
print(f"  R²:   {test_r2:.3f}")
print(f"  bias: {test_bias:.2f} EUR/MWh")

rmse_per_h = np.sqrt(np.mean((test_preds - test_actuals) ** 2, axis=0))
mae_per_h  = np.mean(np.abs(test_preds - test_actuals), axis=0)

print("\nHorizon-specific test metrics:")
for h in [1, 24, 48, 72]:
    print(f"  t+{h:>2}: MAE={mae_per_h[h-1]:>7.2f}  RMSE={rmse_per_h[h-1]:>7.2f} EUR/MWh")


# %% Section 13: Test prediction plot 72 hour forecast 
test_timestamps = (
    data["timestamp"].iloc[cv_hours : cv_hours + FORECAST_H]
    .dt.tz_convert(LOCAL_TIMEZONE)
)
test_pred_flat   = test_preds[0]
test_actual_flat = test_actuals[0]

strategy_label = ("direct" if PRED_STEPS == FORECAST_H
                  else f"recursive {PRED_STEPS}h")

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(test_timestamps, test_actual_flat, label="Actual",
        linewidth=2.5, color="tab:blue")
ax.plot(test_timestamps, test_pred_flat,   label=f"BiLSTM ({strategy_label})",
        linewidth=2.5, color="tab:orange", linestyle="--")

ax.text(0.02, 0.98,
        f"MAE={test_mae:.2f}\nRMSE={test_rmse:.2f}\nR²={test_r2:.3f}\nbias={test_bias:.2f}",
        transform=ax.transAxes, va="top",
        bbox=dict(facecolor="white", edgecolor="0.25", boxstyle="round,pad=0.3"))

ax.set_title(f"Test Set: First {FORECAST_H} h Forecast — Predicted vs Actual")
ax.set_ylabel("EUR/MWh")
ax.legend(loc="lower right")
ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%m-%d %H", tz=test_timestamps.dt.tz)
)
fig.autofmt_xdate(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# %% Section 14: Full 96 h test window — stitched forecast

ORIGIN_2 = 24    # second forecast issued 24 h after the first

full_timestamps = (
    data["timestamp"].iloc[cv_hours : cv_hours + TEST_H]
    .dt.tz_convert(LOCAL_TIMEZONE)
)
actual_full   = price_raw[cv_hours : cv_hours + TEST_H]
stitched_pred = np.concatenate([test_preds[0],
                                test_preds[ORIGIN_2][-ORIGIN_2:]])


stitch_mae, stitch_rmse, stitch_r2, stitch_bias = regression_metrics(
    stitched_pred, actual_full)

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(full_timestamps, actual_full, label="Actual",
        linewidth=2.5, color="tab:blue")
ax.plot(full_timestamps, stitched_pred, label=f"BiLSTM",
        linewidth=2.5, color="tab:orange")

ax.text(0.02, 0.98,
        f"MAE={stitch_mae:.2f}\nRMSE={stitch_rmse:.2f}\n"
        f"R²={stitch_r2:.3f}\nbias={stitch_bias:.2f}",
        transform=ax.transAxes, va="top",
        bbox=dict(facecolor="white", edgecolor="0.25", boxstyle="round,pad=0.3"))

ax.set_title("Final unseen 96 hours: predicted vs actual Dutch day-ahead prices")
ax.set_ylabel("Price (EUR/MWh)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim(full_timestamps.iloc[0],
            full_timestamps.iloc[-1] + pd.Timedelta(hours=1))   # end at 05-29 00
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%m-%d %H", tz=full_timestamps.dt.tz)
)
fig.autofmt_xdate(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# %%
