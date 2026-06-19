# Data pre-processing

Pipeline that turns raw ENTSO-E and weather API exports into the cleaned
regime_B_clean.csv used by all three model folders (LSTM algorithm, Bi-LSTM algorithm,
FFNN_Shah).

## Pipeline order

- Scripts/00_wind_investigation.py - sanity-check wind forecast values per month
- Scripts/01_inspector.py - inspect raw monthly JSON exports
- Scripts/02_parser.py - parse raw JSON into per-series parquet files
- Scripts/03_merger.py - merge parsed series into one table
- Scripts/04_cleaner.py - select final columns, write regime_B_clean.csv
- Scripts/05_features_split.py - feature engineering and train/val/test split (npy/parquet)
- Scripts/06_visualization.py - EDA figures (saved to Results/figures/)

Scripts 00-03 expect raw monthly JSON under Data/<YYYY-MM>/, which isn't included in
this repo since the raw API exports are too large. Scripts 04-06 read/write
output_regimeB/ next to Scripts/. Results/ already contains the finished output
(regime_B_clean.csv and figures) from a previous run, so you don't need the raw data
unless you want to reproduce it from scratch.

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

To reproduce from raw data (requires a Data/ folder with monthly JSON exports):

```bash
python Scripts/00_wind_investigation.py
python Scripts/01_inspector.py
python Scripts/02_parser.py
python Scripts/03_merger.py
python Scripts/04_cleaner.py
python Scripts/05_features_split.py
python Scripts/06_visualization.py
```

Each script writes into output_regimeB/ (created automatically) relative to this
folder.
