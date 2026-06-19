from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import copy_original, load_raw
from src.features import engineer_features, feature_columns, split_final_test
from src.modeling import final_train_and_predict, set_seed, walk_forward_validation
from src.paths import ENGINEERED_CSV, TEST_CSV, TRAIN_CSV, ensure_dirs
from src.plots import plot_final_results, plot_walk_forward


def main() -> None:
    set_seed()
    ensure_dirs()

    copy_original()
    raw = load_raw()
    engineered = engineer_features(raw)
    train_df, test_df = split_final_test(engineered, raw)
    cols = feature_columns(engineered)

    engineered.to_csv(ENGINEERED_CSV, index=False)
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    wf_metrics, wf_preds, wf_histories, best_config, epochs = walk_forward_validation(train_df, cols)
    final_preds, final_metrics = final_train_and_predict(train_df, test_df, cols, best_config, epochs)

    plot_walk_forward(wf_metrics, wf_preds, wf_histories)
    plot_final_results(final_preds, final_metrics)

    print("Pipeline complete.")
    print(f"Rows: raw={len(raw)}, engineered={len(engineered)}, train={len(train_df)}, final_test={len(test_df)}")
    print(f"Features: {len(cols)}")
    print(f"Selected model: {best_config.name}; final epochs={epochs}")


if __name__ == "__main__":
    main()
