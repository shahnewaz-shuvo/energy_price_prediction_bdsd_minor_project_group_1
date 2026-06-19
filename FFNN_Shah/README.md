# FFNN_Shah

Feed-forward neural network forecasting Dutch day-ahead electricity price. This is the
most complete of the three model folders: a proper src/ package, walk-forward CV, a
final 4-day held-out test, and saved model artefacts.

## Layout

- `src/` - config.py (model configs, target, excluded columns), paths.py, data.py
  (load/copy raw data), features.py (feature engineering, train/test split),
  modeling.py (train/eval), metrics.py, plots.py
- `scripts/` - entry points, run_final_4day_pipeline.py is the main one
- `datasets/` - original/ (raw), feature_engineered/, splits/ (train/test)
- `models/` - saved .pt model and .joblib scalers/imputer
- `results/` - walk-forward and final-test metrics, predictions, plots

## Setup

Requires Python 3.10-3.12, see root [README](../README.md#prerequisites). Open a
terminal in this folder, then:

PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Command Prompt:
```bat
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Run

```bash
python -m scripts.run_final_4day_pipeline
```

This runs the full pipeline: copies/loads the raw CSV (already present under
datasets/original/, so this step is skipped if it's already there), engineers
features, splits train/4-day-test, runs walk-forward CV across the configs in
src/config.py (MODEL_CONFIGS), trains the final model, and writes metrics/plots
to results/.

Other scripts under scripts/ (grid_search_heatmap.py, walk_forward_diagram.py)
regenerate specific figures from existing results. Run them the same way, e.g.
`python -m scripts.grid_search_heatmap`.
