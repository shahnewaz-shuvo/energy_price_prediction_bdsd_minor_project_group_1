from __future__ import annotations

import random

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import FINAL_TEST_ROWS, MIN_WALK_FORWARD_TRAIN_ROWS, MODEL_CONFIGS, SEED, TARGET, ModelConfig
from .metrics import metric_dict
from .paths import FEATURE_SCALER_PATH, FINAL_DIR, IMPUTER_PATH, MODEL_PATH, TARGET_SCALER_PATH, WF_DIR


class FFNN(nn.Module):
    def __init__(self, n_features: int, hidden_layers: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = n_features
        for hidden in hidden_layers:
            layers.extend([nn.Linear(in_features, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(dropout)])
            in_features = hidden
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    torch.use_deterministic_algorithms(False)


def build_loss(name: str) -> nn.Module:
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    if name == "mae":
        return nn.L1Loss()
    return nn.MSELoss()


def transform_data(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SimpleImputer, StandardScaler, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(imputer.fit_transform(train_df[cols]))
    y_train = y_scaler.fit_transform(train_df[[TARGET]]).ravel()
    x_eval = x_scaler.transform(imputer.transform(eval_df[cols]))
    y_eval = eval_df[TARGET].to_numpy(dtype=float)
    return x_train, y_train, x_eval, y_eval, imputer, x_scaler, y_scaler


def train_torch_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None,
    y_val_scaled: np.ndarray | None,
    config: ModelConfig,
    fixed_epochs: int | None = None,
) -> tuple[FFNN, pd.DataFrame, int]:
    model = FFNN(x_train.shape[1], list(config.hidden_layers), config.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = build_loss(config.loss)
    dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=min(128, len(dataset)), shuffle=True)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, (fixed_epochs or config.max_epochs) + 1):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(batch_losses))

        val_loss = np.nan
        if x_val is not None and y_val_scaled is not None:
            model.eval()
            with torch.no_grad():
                pred_val = model(torch.tensor(x_val, dtype=torch.float32))
                val_loss = float(criterion(pred_val, torch.tensor(y_val_scaled, dtype=torch.float32)).cpu())
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= config.patience:
                history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
                break
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = fixed_epochs or config.max_epochs
    return model, pd.DataFrame(history), best_epoch


def predict_inverse(model: FFNN, x: np.ndarray, y_scaler: StandardScaler) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(x, dtype=torch.float32)).numpy().reshape(-1, 1)
    return y_scaler.inverse_transform(pred_scaled).ravel()


def fit_ridge_baseline(train_df: pd.DataFrame, eval_df: pd.DataFrame, cols: list[str]) -> tuple[Ridge, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train_df[cols]))
    x_eval = scaler.transform(imputer.transform(eval_df[cols]))
    model = Ridge(alpha=10.0)
    model.fit(x_train, train_df[TARGET].to_numpy(dtype=float))
    return model, model.predict(x_eval)


def build_walk_forward_folds(
    n_rows: int,
    min_train_rows: int = MIN_WALK_FORWARD_TRAIN_ROWS,
    horizon_rows: int = FINAL_TEST_ROWS,
) -> list[tuple[int, int, int]]:
    n_folds = (n_rows - min_train_rows) // horizon_rows
    if n_folds < 1:
        raise ValueError(
            f"Not enough rows for walk-forward validation: n_rows={n_rows}, "
            f"min_train_rows={min_train_rows}, horizon_rows={horizon_rows}"
        )
    first_val_start = n_rows - n_folds * horizon_rows
    return [
        (fold_num, first_val_start + (fold_num - 1) * horizon_rows, first_val_start + fold_num * horizon_rows)
        for fold_num in range(1, n_folds + 1)
    ]


def walk_forward_validation(train_df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ModelConfig, int]:
    folds = build_walk_forward_folds(len(train_df))

    config_scores = []
    for config in MODEL_CONFIGS:
        fold_mae, fold_rmse, fold_r2, fold_bias = [], [], [], []
        for fold_num, val_start, val_end in folds:
            set_seed(SEED + fold_num)
            fold_train = train_df.iloc[:val_start].copy()
            fold_val = train_df.iloc[val_start:val_end].copy()
            x_train, y_train_scaled, x_val, y_val, _, _, y_scaler = transform_data(fold_train, fold_val, cols)
            y_val_scaled = y_scaler.transform(fold_val[[TARGET]]).ravel()
            model, _, _ = train_torch_model(x_train, y_train_scaled, x_val, y_val_scaled, config)
            metrics = metric_dict(y_val, predict_inverse(model, x_val, y_scaler))
            fold_mae.append(metrics["MAE"])
            fold_rmse.append(metrics["RMSE"])
            fold_r2.append(metrics["R2"])
            fold_bias.append(metrics["bias"])
        config_scores.append(
            {
                "config": config.name,
                "fold_count": len(folds),
                "mean_val_MAE": float(np.mean(fold_mae)),
                "std_val_MAE": float(np.std(fold_mae, ddof=1)) if len(fold_mae) > 1 else 0.0,
                "min_val_MAE": float(np.min(fold_mae)),
                "max_val_MAE": float(np.max(fold_mae)),
                "mean_val_RMSE": float(np.mean(fold_rmse)),
                "mean_val_R2": float(np.mean(fold_r2)),
                "mean_val_bias": float(np.mean(fold_bias)),
            }
        )

    best_config = next(c for c in MODEL_CONFIGS if c.name == min(config_scores, key=lambda d: d["mean_val_MAE"])["config"])
    all_metrics, all_predictions, histories, best_epochs = [], [], [], []
    for fold_num, val_start, val_end in folds:
        set_seed(SEED + fold_num)
        fold_train = train_df.iloc[:val_start].copy()
        fold_val = train_df.iloc[val_start:val_end].copy()
        x_train, y_train_scaled, x_val, y_val, _, _, y_scaler = transform_data(fold_train, fold_val, cols)
        y_val_scaled = y_scaler.transform(fold_val[[TARGET]]).ravel()
        model, history, best_epoch = train_torch_model(x_train, y_train_scaled, x_val, y_val_scaled, best_config)
        pred = predict_inverse(model, x_val, y_scaler)
        _, ridge_pred = fit_ridge_baseline(fold_train, fold_val, cols)

        for model_name, model_pred in [
            ("FFNN", pred),
            ("Ridge", ridge_pred),
            ("Lag 168h", fold_val["price_lag_168h"].to_numpy(dtype=float)),
            ("Lag 96h", fold_val["price_lag_96h"].to_numpy(dtype=float)),
        ]:
            metrics = metric_dict(y_val, model_pred)
            metrics.update(
                {
                    "fold": fold_num,
                    "model": model_name,
                    "validation_start_local": str(fold_val["timestamp_local"].iloc[0]),
                    "validation_end_local": str(fold_val["timestamp_local"].iloc[-1]),
                }
            )
            all_metrics.append(metrics)

        fold_pred = fold_val[["timestamp", "timestamp_local", "local_date", "local_hour", TARGET]].copy()
        fold_pred["prediction_ffnn"] = pred
        fold_pred["prediction_ridge"] = ridge_pred
        fold_pred["baseline_lag_168h"] = fold_val["price_lag_168h"].to_numpy(dtype=float)
        fold_pred["baseline_lag_96h"] = fold_val["price_lag_96h"].to_numpy(dtype=float)
        fold_pred["fold"] = fold_num
        all_predictions.append(fold_pred)
        history["fold"] = fold_num
        histories.append(history)
        best_epochs.append(best_epoch)

    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.concat(all_predictions, ignore_index=True)
    histories_df = pd.concat(histories, ignore_index=True)
    pd.DataFrame(config_scores).to_csv(WF_DIR / "model_selection_scores.csv", index=False)
    metrics_df.to_csv(WF_DIR / "walk_forward_metrics.csv", index=False)
    preds_df.to_csv(WF_DIR / "predictions" / "walk_forward_predictions.csv", index=False)
    histories_df.to_csv(WF_DIR / "walk_forward_loss_history.csv", index=False)
    return metrics_df, preds_df, histories_df, best_config, max(20, int(np.median(best_epochs)))


def final_train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
    config: ModelConfig,
    epochs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(SEED + 10_000)
    x_train, y_train_scaled, x_test, y_test, imputer, x_scaler, y_scaler = transform_data(train_df, test_df, cols)
    model, final_history, _ = train_torch_model(x_train, y_train_scaled, None, None, config, fixed_epochs=epochs)
    pred = predict_inverse(model, x_test, y_scaler)
    _, ridge_pred = fit_ridge_baseline(train_df, test_df, cols)

    result = test_df[["timestamp", "timestamp_local", "local_date", "local_hour", TARGET]].copy()
    result["prediction_ffnn"] = pred
    result["prediction_ridge"] = ridge_pred
    result["baseline_lag_168h"] = test_df["price_lag_168h"].to_numpy(dtype=float)
    result["baseline_lag_96h"] = test_df["price_lag_96h"].to_numpy(dtype=float)
    result["residual_ffnn"] = result["prediction_ffnn"] - result[TARGET]
    result["abs_error_ffnn"] = result["residual_ffnn"].abs()

    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": cols,
            "model_config": config.__dict__,
            "epochs": epochs,
            "target": TARGET,
        },
        MODEL_PATH,
    )
    joblib.dump(x_scaler, FEATURE_SCALER_PATH)
    joblib.dump(y_scaler, TARGET_SCALER_PATH)
    joblib.dump(imputer, IMPUTER_PATH)
    result.to_csv(FINAL_DIR / "predictions" / "final_96h_predictions.csv", index=False)
    final_history.to_csv(FINAL_DIR / "predictions" / "final_training_loss_history.csv", index=False)

    rows = []
    actual = result[TARGET].to_numpy(dtype=float)
    for model_name, col in [
        ("FFNN", "prediction_ffnn"),
        ("Ridge", "prediction_ridge"),
        ("Lag 168h", "baseline_lag_168h"),
        ("Lag 96h", "baseline_lag_96h"),
    ]:
        row = metric_dict(actual, result[col].to_numpy(dtype=float))
        row["model"] = model_name
        row["scope"] = "final_96h"
        rows.append(row)
    for local_date, day_df in result.groupby("local_date"):
        row = metric_dict(day_df[TARGET].to_numpy(dtype=float), day_df["prediction_ffnn"].to_numpy(dtype=float))
        row["model"] = "FFNN"
        row["scope"] = f"day_{local_date}"
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(FINAL_DIR / "final_metrics.csv", index=False)
    return result, metrics_df
