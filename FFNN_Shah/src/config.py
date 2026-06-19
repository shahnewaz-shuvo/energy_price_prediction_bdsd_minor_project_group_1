from __future__ import annotations

from dataclasses import dataclass


SEED = 42
TARGET = "price_entsoe"
LOCAL_TZ = "Europe/Amsterdam"
FINAL_TEST_ROWS = 96
MIN_WALK_FORWARD_TRAIN_ROWS = 1000


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden_layers: tuple[int, ...]
    dropout: float
    lr: float
    weight_decay: float
    loss: str
    max_epochs: int = 260
    patience: int = 28


MODEL_CONFIGS = [
    ModelConfig("ffnn_huber_128_64", (128, 64), 0.15, 0.001, 1e-4, "huber"),
    ModelConfig("ffnn_mse_128_64", (128, 64), 0.001, 0.001, 1e-4, "mse"),
    ModelConfig("ffnn_huber_64_32", (64, 32), 0.10, 0.0015, 1e-4, "huber"),
]

EXCLUDED_MODEL_COLUMNS = {
    "timestamp",
    "timestamp_local",
    "local_date",
    "source_month",
    "data_type",
    "is_forecast_month",
    TARGET,
    "gen_FR_nuclear_actual_mw",
    "grid_balance_delta_mw",
    "continuous_segment_id",
}
