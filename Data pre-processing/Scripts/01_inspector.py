# Script 01 - inspector
# opens one sample file of each type and prints what's inside, so we know


import json
import os

# go up one level to find the Data folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")

TARGET_MONTHS = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]

# the file types we expect each day
FILE_TYPES = [
    "energy_price_forecast",
    "weather_forecast_multi_location",
    "wind_forecast",
    "solar_forecast",
    "demand_weather_forecast",
    "calendar_features",
    "cross_border_flows",
    "load_forecast",
    "generation_forecast",
    "market_proxies",
    "ned_production",
    "grid_imbalance",
    "gas_flows",
    "gas_storage",
]


def find_first_file(month_folder_path, file_type_keyword):
    # first json file in the folder whose name contains the keyword
    try:
        all_files = sorted(os.listdir(month_folder_path))
    except FileNotFoundError:
        return None

    for fname in all_files:
        if file_type_keyword in fname and fname.endswith(".json"):
            return os.path.join(month_folder_path, fname)
    return None


def safe_open_json(filepath):
    # returns (data, error). error is None if it parsed fine.
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        if not raw or raw in ("{}", "[]", "null"):
            return None, "EMPTY FILE"

        # encrypted files don't start with a json character
        if not raw[0] in ("{", "[", '"'):
            return None, f"ENCRYPTED/CORRUPT - starts with: {raw[:30]!r}"

        data = json.loads(raw)
        return data, None

    except json.JSONDecodeError as e:
        return None, f"JSON PARSE ERROR - {e}"
    except Exception as e:
        return None, f"UNEXPECTED ERROR - {e}"


def describe_structure(data, depth=0, max_depth=3):
    # prints the keys and value types without dumping all the data
    indent = "    " * depth

    if depth > max_depth:
        print(f"{indent}... (truncated at depth {max_depth})")
        return

    if isinstance(data, dict):
        if len(data) == 0:
            print(f"{indent}(empty dict)")
            return
        keys = list(data.keys())
        # if the keys look like timestamps, just summarise instead of listing all
        sample_key = keys[0]
        if "T" in str(sample_key) and ("+" in str(sample_key) or "Z" in str(sample_key)):
            print(f"{indent}Timestamp-keyed dict with {len(keys)} entries")
            print(f"{indent}  First key : {keys[0]}")
            print(f"{indent}  Last key  : {keys[-1]}")
            sample_val = data[keys[0]]
            print(f"{indent}  Value type: {type(sample_val).__name__}")
            if isinstance(sample_val, (int, float)):
                print(f"{indent}  Example   : {sample_val}")
            elif isinstance(sample_val, dict):
                print(f"{indent}  Value keys: {list(sample_val.keys())[:8]}")
        else:
            for key in keys[:10]:
                val = data[key]
                print(f"{indent}['{key}'] -> {type(val).__name__}")
                if isinstance(val, (dict, list)) and depth < max_depth:
                    describe_structure(val, depth + 1, max_depth)
                elif isinstance(val, (int, float, str, bool)):
                    display = str(val)[:80]
                    print(f"{indent}    = {display}")
            if len(keys) > 10:
                print(f"{indent}  ... and {len(keys) - 10} more keys")

    elif isinstance(data, list):
        if len(data) == 0:
            print(f"{indent}(empty list)")
            return
        print(f"{indent}List with {len(data)} items. First item:")
        describe_structure(data[0], depth + 1, max_depth)

    else:
        print(f"{indent}{type(data).__name__}: {str(data)[:80]}")


print("=" * 70)
print("SCRIPT 01 - INSPECTOR")
print("=" * 70)
print(f"Data folder : {DATA_DIR}")
print(f"Months      : {TARGET_MONTHS}")
print(f"File types  : {len(FILE_TYPES)} types to check")
print("=" * 70)

# part 1 - which files exist in each month
print("\n" + "-" * 70)
print("PART 1 - FILE AVAILABILITY")
print("-" * 70)
print(f"{'File Type':<40} " + " ".join(f"{m[-5:]:>7}" for m in TARGET_MONTHS))
print("-" * 70)

availability = {}
for ftype in FILE_TYPES:
    row = f"{ftype:<40} "
    availability[ftype] = {}
    for month in TARGET_MONTHS:
        month_path = os.path.join(DATA_DIR, month)
        found = find_first_file(month_path, ftype)
        if found:
            row += f"{'YES':>7} "
            availability[ftype][month] = found
        else:
            row += f"{'---':>7} "
            availability[ftype][month] = None
    print(row)

# part 2 - look inside one file of each type
print("\n" + "-" * 70)
print("PART 2 - STRUCTURE (one sample file per type)")
print("-" * 70)

for ftype in FILE_TYPES:
    # use the first month that has this file type
    sample_path = None
    sample_month = None
    for month in TARGET_MONTHS:
        if availability[ftype].get(month):
            sample_path = availability[ftype][month]
            sample_month = month
            break

    if sample_path is None:
        print(f"\n[{ftype}]")
        print("  not found in any month - skip")
        continue

    print(f"\n[{ftype}]")
    print(f"  Sample file : {os.path.basename(sample_path)}")
    print(f"  From month  : {sample_month}")

    data, error = safe_open_json(sample_path)

    if error:
        print(f"  STATUS      : FAILED - {error}")
        continue

    print(f"  STATUS      : OK")
    print(f"  Top-level type: {type(data).__name__}")
    print("  STRUCTURE:")
    describe_structure(data, depth=2)

# part 3 - file count per month
print("\n" + "-" * 70)
print("PART 3 - FILE COUNT PER MONTH")
print("-" * 70)

for month in TARGET_MONTHS:
    month_path = os.path.join(DATA_DIR, month)
    if not os.path.exists(month_path):
        print(f"  {month}: FOLDER NOT FOUND")
        continue
    all_files = [f for f in os.listdir(month_path) if f.endswith(".json")]
    print(f"  {month}: {len(all_files)} JSON files")

print("\n" + "=" * 70)
print("=" * 70)
