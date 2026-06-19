# Script 00 - wind check
# opens one wind_forecast file per month and prints the real values inside,


import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")

TARGET_MONTHS = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]

def safe_load_json(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw or raw[0] not in ("{", "[", '"'):
            return None
        return json.loads(raw)
    except Exception:
        return None

def get_first_file(month, keyword):
    folder = os.path.join(DATA_DIR, month)
    if not os.path.exists(folder):
        return None
    matches = sorted([f for f in os.listdir(folder)
                      if keyword in f and f.endswith(".json")])
    return os.path.join(folder, matches[0]) if matches else None

def print_dict_sample(d, label, n=5):
    # print the first few timestamp -> value pairs
    if not isinstance(d, dict) or len(d) == 0:
        print(f"      {label}: EMPTY")
        return
    print(f"      {label} ({len(d)} entries total, showing first {n}):")
    for i, (k, v) in enumerate(d.items()):
        if i >= n:
            break
        print(f"        {k}  ->  {v}")

print("=" * 70)
print("WIND FORECAST CHECK - real values per month")
print("=" * 70)

for month in TARGET_MONTHS:
    filepath = get_first_file(month, "wind_forecast")
    print(f"\n{'-'*70}")
    print(f"MONTH: {month}")

    if not filepath:
        print("  No wind_forecast file found in this month")
        continue

    print(f"  File: {os.path.basename(filepath)}")
    d = safe_load_json(filepath)

    if not d:
        print("  Could not parse file")
        continue

    # go through each top-level key and print its values
    for top_key, top_val in d.items():
        if top_key == "version":
            print(f"  version: {top_val}")
            continue
        if not isinstance(top_val, dict):
            continue

        print(f"\n  [{top_key}]")

        # metadata usually tells us the unit
        metadata = top_val.get("metadata", {})
        if metadata:
            for meta_field in ("units", "data_type", "source", "description"):
                if meta_field in metadata:
                    print(f"    metadata.{meta_field}: {metadata[meta_field]}")

        data = top_val.get("data", {})
        print_dict_sample(data, "data values")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
