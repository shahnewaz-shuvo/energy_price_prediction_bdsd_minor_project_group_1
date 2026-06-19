from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = PROJECT_ROOT.parent / "regime_B_full_data.csv"
ORIGINAL_CSV = PROJECT_ROOT / "datasets" / "original" / "regime_B_full_data.csv"
ENGINEERED_CSV = PROJECT_ROOT / "datasets" / "feature_engineered" / "full_data_engineered.csv"
TRAIN_CSV = PROJECT_ROOT / "datasets" / "splits" / "final_4day_train.csv"
TEST_CSV = PROJECT_ROOT / "datasets" / "splits" / "final_4day_test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "final_4day_ffnn_model.pt"
FEATURE_SCALER_PATH = PROJECT_ROOT / "models" / "feature_scaler.joblib"
TARGET_SCALER_PATH = PROJECT_ROOT / "models" / "target_scaler.joblib"
IMPUTER_PATH = PROJECT_ROOT / "models" / "feature_imputer.joblib"
WF_DIR = PROJECT_ROOT / "results" / "walk_forward"
FINAL_DIR = PROJECT_ROOT / "results" / "final_4day_test"
REPORT_DIR = PROJECT_ROOT / "results" / "report"


def ensure_dirs() -> None:
    for path in [
        ORIGINAL_CSV.parent,
        ENGINEERED_CSV.parent,
        TRAIN_CSV.parent,
        MODEL_PATH.parent,
        WF_DIR / "graphs",
        WF_DIR / "predictions",
        FINAL_DIR / "graphs",
        FINAL_DIR / "predictions",
        REPORT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
