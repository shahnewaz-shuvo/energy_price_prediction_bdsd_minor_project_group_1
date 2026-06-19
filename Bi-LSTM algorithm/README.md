# Bi-LSTM algorithm

Bidirectional LSTM (with a learned forward/backward attention blend) forecasting Dutch
day-ahead electricity price 72 hours ahead, using regime_B_features_selected.csv.

Same walk-forward CV and Optuna search structure as LSTM algorithm/, run as a plain
Python script (cell-delimited with # %% markers, so it can also be opened in VS Code's
Python Interactive window).

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
python Bi_LSTM.py
```

CSV_PATH resolves relative to this script's location, so it works regardless of your
current working directory. Plots are shown interactively (plt.show()); save them
yourself if running headless.
