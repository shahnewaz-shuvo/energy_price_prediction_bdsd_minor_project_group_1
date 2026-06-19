# LSTM algorithm

Sequence-to-sequence LSTM that forecasts Dutch day-ahead electricity price
(price_entsoe) 72 hours ahead, using regime_B_clean.csv (from Data pre-processing/).

Walk-forward cross-validation (20 expanding folds) with Optuna hyperparameter search,
then a held-out 96 h test evaluation.

## Setup

Requires Python 3.10-3.12, see root [README](../README.md#prerequisites). Open a
terminal in this folder, then:

PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name lstm-algorithm
```

Command Prompt:
```bat
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python -m ipykernel install --user --name lstm-algorithm
```

## Run

Open LSTM,Final,_NEW.ipynb in Jupyter/VS Code, select the lstm-algorithm kernel, and
run all cells top to bottom. CSV_PATH is set to regime_B_clean.csv and resolves
relative to the notebook's working directory, so run it from inside this folder
(Jupyter defaults to that when you open the notebook here).

This notebook originally targeted Google Colab. The Colab-only
google.colab.files.download() call has been replaced with a local save so it runs
on any machine.

Output: optimisation diagnostics, walk-forward CV metrics/plots, and a final 96 h
predicted-vs-actual forecast, all saved to plots/.
