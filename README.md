# energy_price_prediction_bdsd_minor_project_group_1

This project is part of a larger study on electricity price forecasting for hydrogen system
operation. The aim is to find a suitable neural network approach to predict electricity
prices for up to 3 days ahead, which can help improve hydrogen production scheduling.

Three model approaches trained and evaluated on the same dataset (regime B), built through
a shared pre-processing pipeline.

## Structure

- [Data pre-processing/](Data%20pre-processing/README.md) - cleans raw data into `regime_B_clean.csv`
- [LSTM algorithm/](LSTM%20algorithm/README.md) - uni-directional LSTM, Jupyter notebook
- [Bi-LSTM algorithm/](Bi-LSTM%20algorithm/README.md) - bidirectional LSTM, plain script
- [FFNN_algorithm/](FFNN_algorithm/README.md) - feed-forward NN, full src/ package and saved artefacts

Each model folder has its own README.md and requirements.txt. They're independent
(separate venvs), since each one uses a different library mix and dataset snapshot.

## Prerequisites

Install Python 3.10, 3.11, or 3.12 from [python.org/downloads](https://www.python.org/downloads/),
and tick "Add python.exe to PATH" during install. Newer versions (3.13+) may not have a
`torch` package available yet, which would make `pip install` fail.

Check it worked by opening a terminal and running `python --version`.

## Quick start

Pick a folder, open a terminal in that folder, and run:

PowerShell (default on Windows 11):
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

If PowerShell blocks the activation script with a "running scripts is disabled" error,
run this once and try again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then check that folder's README for the run command.

## Target and features

All three models forecast `price_entsoe` (EUR/MWh) using load forecasts, cross-border
flows, weather (humidity, temperature, wind, solar), and cyclical time-of-day/week
features. Validation uses expanding-window walk-forward cross-validation with a final
held-out test window.
