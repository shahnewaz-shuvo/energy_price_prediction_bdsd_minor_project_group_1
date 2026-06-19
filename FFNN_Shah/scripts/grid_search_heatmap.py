from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import SEED, TARGET
from src.features import feature_columns
from src.modeling import set_seed
from src.paths import TRAIN_CSV, ensure_dirs

HIDDEN_SIZES = [64, 128, 256]
DROPOUTS = [0.1, 0.2, 0.3]
NUM_LAYERS_LIST = [1, 2]
VAL_FRAC = 0.20
MAX_EPOCHS = 120
PATIENCE = 15
BATCH_SIZE = 128
LR = 0.001
WEIGHT_DECAY = 1e-4


def _build(n_features: int, hidden_size: int, num_layers: int, dropout: float) -> nn.Module:
    layers: list[nn.Module] = []
    in_f = n_features
    for _ in range(num_layers):
        layers += [nn.Linear(in_f, hidden_size), nn.ReLU(), nn.BatchNorm1d(hidden_size), nn.Dropout(dropout)]
        in_f = hidden_size
    layers.append(nn.Linear(in_f, 1))
    return nn.Sequential(*layers)


def _train_eval(x_tr, y_tr, x_va, y_va, hidden_size, num_layers, dropout, seed):
    set_seed(seed)
    model = _build(x_tr.shape[1], hidden_size, num_layers, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(x_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=min(BATCH_SIZE, len(x_tr)),
        shuffle=True,
    )
    xv = torch.tensor(x_va, dtype=torch.float32)
    yv = torch.tensor(y_va, dtype=torch.float32)
    best_val, stale = float("inf"), 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb).squeeze(1), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_mse = float(crit(model(xv).squeeze(1), yv).cpu())
        if val_mse < best_val - 1e-6:
            best_val, stale = val_mse, 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    return best_val


def main() -> None:
    ensure_dirs()
    set_seed(SEED)

    train_df = pd.read_csv(TRAIN_CSV, parse_dates=["timestamp"])
    cols = feature_columns(train_df)

    n_val = int(len(train_df) * VAL_FRAC)
    tr, va = train_df.iloc[: len(train_df) - n_val], train_df.iloc[len(train_df) - n_val :]

    imputer = SimpleImputer(strategy="median")
    xs = StandardScaler()
    ys = StandardScaler()
    x_tr = xs.fit_transform(imputer.fit_transform(tr[cols]))
    y_tr = ys.fit_transform(tr[[TARGET]]).ravel()
    x_va = xs.transform(imputer.transform(va[cols]))
    y_va = ys.transform(va[[TARGET]]).ravel()

    results: dict[tuple, float] = {}
    best_mse, best_combo = float("inf"), None
    total = len(NUM_LAYERS_LIST) * len(HIDDEN_SIZES) * len(DROPOUTS)
    done = 0
    for nl in NUM_LAYERS_LIST:
        for hs in HIDDEN_SIZES:
            for dr in DROPOUTS:
                done += 1
                print(f"[{done}/{total}] layers={nl}, hidden={hs}, dropout={dr} ...", end=" ", flush=True)
                mse = _train_eval(x_tr, y_tr, x_va, y_va, hs, nl, dr, SEED)
                results[(nl, hs, dr)] = mse
                print(f"{mse:.4f}")
                if mse < best_mse:
                    best_mse, best_combo = mse, (nl, hs, dr)

    all_vals = list(results.values())
    vmin, vmax = min(all_vals), max(all_vals)
    cmap = plt.cm.RdYlGn_r

    fig, axes = plt.subplots(1, len(NUM_LAYERS_LIST), figsize=(12, 5), sharey=True)
    im = None
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for ax, nl in zip(axes, NUM_LAYERS_LIST):
        matrix = np.array([[results[(nl, hs, dr)] for hs in HIDDEN_SIZES] for dr in DROPOUTS])
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(HIDDEN_SIZES)))
        ax.set_xticklabels(HIDDEN_SIZES)
        ax.set_yticks(range(len(DROPOUTS)))
        ax.set_yticklabels(DROPOUTS)
        ax.set_xlabel("Hidden size")
        ax.set_ylabel("Dropout")
        ax.set_title(f"Num layers = {nl}")

        legend_added = False
        for i, dr in enumerate(DROPOUTS):
            for j, hs in enumerate(HIDDEN_SIZES):
                val = results[(nl, hs, dr)]
                rgba = cmap(norm(val))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if lum < 0.55 else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=9, color=text_color)
                if (nl, hs, dr) == best_combo:
                    rect = patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        linewidth=2.5, edgecolor="#0000cc", facecolor="none",
                    )
                    ax.add_patch(rect)
                    if not legend_added:
                        ax.plot([], [], color="#0000cc", linewidth=2.5, label="Best combination")
                        ax.legend(loc="upper right", fontsize=8)
                        legend_added = True

    fig.colorbar(im, ax=axes[-1], label="Val MSE")
    fig.suptitle("Grid search: Val MSE by hyperparameter combination")
    plt.tight_layout()

    out = PROJECT_ROOT / "results" / "grid_search_heatmap.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")
    print(f"Best: layers={best_combo[0]}, hidden={best_combo[1]}, dropout={best_combo[2]}, Val MSE={best_mse:.4f}")


if __name__ == "__main__":
    main()
