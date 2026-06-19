# Script 02 - parser
# reads all the json files and pulls each data type into its own parquet table.

import json
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# paths
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_regimeB")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_MONTHS   = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
FORECAST_MONTHS = {"2026-03", "2026-04", "2026-05"}   # these months are forecast data

# helpers

def safe_load_json(filepath):
    # open a json file, return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw or raw in ("{}", "[]", "null"):
            return None
        if raw[0] not in ("{", "[", '"'):
            return None  # encrypted
        return json.loads(raw)
    except Exception:
        return None

def get_files(month, keyword):
    # all json files in a month folder whose name has the keyword
    folder = os.path.join(DATA_DIR, month)
    if not os.path.exists(folder):
        return []
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if keyword in f and f.endswith(".json")
    ])

def to_utc(ts_str):
    # turn any timestamp string into a UTC timestamp
    try:
        return pd.to_datetime(ts_str, utc=True)
    except Exception:
        return pd.NaT

def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def meta(month):
    # the two columns every row gets
    return {
        "source_month":      month,
        "is_forecast_month": month in FORECAST_MONTHS,
    }

# energy price: data["entsoe"]["data"] = {timestamp: value}, 

def parse_energy(month, filepath):
    d = safe_load_json(filepath)
    if not d:
        return []

    # Dec uses 'energy_zero', Jan onwards uses 'entsoe_de' and 'elspot'
    key_to_col = {
        "entsoe":       "price_entsoe",
        "entsoe_de":    "price_entsoe_de",
        "energy_zero":  "price_energy_zero",
        "epex":         "price_epex",
        "elspot":       "price_elspot",
    }

    combined = {}
    for key, col in key_to_col.items():
        section = d.get(key, {})
        data_dict = section.get("data", {}) if isinstance(section, dict) else {}
        for ts_str, val in data_dict.items():
            ts = to_utc(ts_str)
            if pd.isna(ts):
                continue
            combined.setdefault(ts, {})[col] = safe_float(val)

    rows = []
    for ts, vals in combined.items():
        row = {"timestamp": ts, **meta(month), **vals}
        rows.append(row)
    return rows

# weather: data["data"][location] = {timestamp: {fields}} - average over the NL locations

def parse_weather(month, filepath):
    d = safe_load_json(filepath)
    if not d or "data" not in d:
        return []

    locations = d["data"]
    if not isinstance(locations, dict):
        return []

    # combined[ts][field] = list of values from each location
    combined = {}

    for loc_name, loc_data in locations.items():
        if not isinstance(loc_data, dict):
            continue
        for ts_str, entry in loc_data.items():
            ts = to_utc(ts_str)
            if pd.isna(ts) or not isinstance(entry, dict):
                continue
            combined.setdefault(ts, {})
            for field, val in entry.items():
                combined[ts].setdefault(field, [])
                fv = safe_float(val)
                if fv is not None:
                    combined[ts][field].append(fv)

    rows = []
    for ts, fields in combined.items():
        row = {"timestamp": ts, **meta(month)}
        # average over the locations
        for field, vals in fields.items():
            if vals:
                row[f"weather_{field}"] = sum(vals) / len(vals)
        rows.append(row)
    return rows

# wind: we take wind_speed_120m (close to offshore hub height) and average over locations

def parse_wind(month, filepath):
    d = safe_load_json(filepath)
    if not d:
        return []

    combined = {}   # timestamp -> list of wind_speed_120m values

    for top_key, top_val in d.items():
        if top_key == "version" or not isinstance(top_val, dict):
            continue
        data_dict = top_val.get("data", top_val)
        if not isinstance(data_dict, dict):
            continue

        for ts_or_loc, val in data_dict.items():
            # case A: value already has wind_speed_120m
            if isinstance(val, dict) and "wind_speed_120m" in val:
                ts = to_utc(ts_or_loc)
                if pd.isna(ts):
                    continue
                spd = safe_float(val.get("wind_speed_120m"))
                if spd is not None:
                    combined.setdefault(ts, []).append(spd)

            # case B: value is a location dict {timestamp: {fields}}
            elif isinstance(val, dict):
                for ts_str, entry in val.items():
                    if not isinstance(entry, dict):
                        continue
                    if "wind_speed_120m" not in entry:
                        continue
                    ts = to_utc(ts_str)
                    if pd.isna(ts):
                        continue
                    spd = safe_float(entry.get("wind_speed_120m"))
                    if spd is not None:
                        combined.setdefault(ts, []).append(spd)

    rows = []
    for ts, speeds in combined.items():
        if speeds:
            rows.append({
                "timestamp":    ts,
                **meta(month),
                "wind_speed_ms": round(sum(speeds) / len(speeds), 4),
            })
    return rows

# solar: same shape as weather - average over locations

def parse_solar(month, filepath):
    d = safe_load_json(filepath)
    if not d or "data" not in d:
        return []

    locations = d["data"]
    if not isinstance(locations, dict):
        return []

    combined = {}
    for loc_name, loc_data in locations.items():
        if not isinstance(loc_data, dict):
            continue
        for ts_str, entry in loc_data.items():
            ts = to_utc(ts_str)
            if pd.isna(ts) or not isinstance(entry, dict):
                continue
            combined.setdefault(ts, {})
            for field, val in entry.items():
                combined[ts].setdefault(field, [])
                fv = safe_float(val)
                if fv is not None:
                    combined[ts][field].append(fv)

    rows = []
    for ts, fields in combined.items():
        row = {"timestamp": ts, **meta(month)}
        for field, vals in fields.items():
            if vals:
                row[f"solar_{field}"] = sum(vals) / len(vals)
        rows.append(row)
    return rows

# demand-weather: 11 cities, average per hour

def parse_demand_weather(month, filepath):
    d = safe_load_json(filepath)
    if not d or "data" not in d:
        return []

    locations = d["data"]
    if not isinstance(locations, dict):
        return []

    combined = {}
    for loc_name, loc_data in locations.items():
        if not isinstance(loc_data, dict):
            continue
        for ts_str, entry in loc_data.items():
            ts = to_utc(ts_str)
            if pd.isna(ts) or not isinstance(entry, dict):
                continue
            combined.setdefault(ts, {})
            for field, val in entry.items():
                combined[ts].setdefault(field, [])
                fv = safe_float(val)
                if fv is not None:
                    combined[ts][field].append(fv)

    rows = []
    for ts, fields in combined.items():
        row = {"timestamp": ts, **meta(month)}
        for field, vals in fields.items():
            if vals:
                row[f"dw_{field}"] = sum(vals) / len(vals)
        rows.append(row)
    return rows

# calendar: data["data"] = {timestamp: {year, month, hour, is_weekend, ...}}

def parse_calendar(month, filepath):
    d = safe_load_json(filepath)
    if not d or "data" not in d:
        return []

    data_dict = d["data"]
    if not isinstance(data_dict, dict):
        return []

    rows = []
    for ts_str, entry in data_dict.items():
        ts = to_utc(ts_str)
        if pd.isna(ts) or not isinstance(entry, dict):
            continue
        row = {"timestamp": ts, **meta(month)}
        for field, val in entry.items():
            row[f"cal_{field}"] = val
        rows.append(row)
    return rows

# cross-border flows: data["data"]["flows"] = {timestamp: {NL_DE: val, NL_BE: val, ...}}

def parse_cross_border(month, filepath):
    d = safe_load_json(filepath)
    if not d:
        return []

    flows = d.get("data", {}).get("flows", {})
    if not isinstance(flows, dict):
        return []

    rows = []
    for ts_str, entry in flows.items():
        ts = to_utc(ts_str)
        if pd.isna(ts) or not isinstance(entry, dict):
            continue
        row = {"timestamp": ts, **meta(month)}
        for border, val in entry.items():
            row[f"flow_{border}_mw"] = safe_float(val)
        rows.append(row)
    return rows

# load forecast: data["data"]["NL"] = {timestamp: {forecast, actual}}, plus DE_LU

def parse_load(month, filepath):
    d = safe_load_json(filepath)
    if not d:
        return []

    zones = d.get("data", {})
    if not isinstance(zones, dict):
        return []

    rows = []
    # one set of columns per zone
    for zone_name, zone_data in zones.items():
        if not isinstance(zone_data, dict):
            continue
        for ts_str, entry in zone_data.items():
            ts = to_utc(ts_str)
            if pd.isna(ts):
                continue
            row = {"timestamp": ts, **meta(month)}
            if isinstance(entry, dict):
                for field, val in entry.items():
                    row[f"load_{zone_name}_{field}_mw"] = safe_float(val)
            else:
                row[f"load_{zone_name}_mw"] = safe_float(entry)
            rows.append(row)
    return rows

# generation: data["data"]["FR"] = {timestamp: {gen_type: val}}

def parse_generation(month, filepath):
    d = safe_load_json(filepath)
    if not d:
        return []

    countries = d.get("data", {})
    if not isinstance(countries, dict):
        return []

    rows = []
    for country, country_data in countries.items():
        if not isinstance(country_data, dict):
            continue
        for ts_str, entry in country_data.items():
            ts = to_utc(ts_str)
            if pd.isna(ts):
                continue
            row = {"timestamp": ts, **meta(month)}
            if isinstance(entry, dict):
                for gen_type, val in entry.items():
                    row[f"gen_{country}_{gen_type}_mw"] = safe_float(val)
            else:
                row[f"gen_{country}_total_mw"] = safe_float(entry)
            rows.append(row)
    return rows

# market proxies (gas / carbon)

def parse_market_proxies(month, filepath):
    d = safe_load_json(filepath)
    if not d:
        return []

    commodities = d.get("data", {})
    if not isinstance(commodities, dict):
        return []

    combined = {}
    for commodity, comm_data in commodities.items():
        if not isinstance(comm_data, dict):
            continue
        for ts_str, entry in comm_data.items():
            ts = to_utc(ts_str)
            if pd.isna(ts):
                continue
            combined.setdefault(ts, {})
            if isinstance(entry, dict):
                for field, val in entry.items():
                    combined[ts][f"market_{commodity}_{field}"] = safe_float(val)
            else:
                combined[ts][f"market_{commodity}_price"] = safe_float(entry)

    rows = []
    for ts, vals in combined.items():
        rows.append({"timestamp": ts, **meta(month), **vals})
    return rows

# ned production (solar / wind onshore / wind offshore)

def parse_ned(month, filepath):
    d = safe_load_json(filepath)
    if not d:
        return []

    energy_types = d.get("data", {})
    if not isinstance(energy_types, dict):
        return []

    combined = {}
    for etype, etype_data in energy_types.items():
        if not isinstance(etype_data, dict):
            continue
        for ts_str, entry in etype_data.items():
            ts = to_utc(ts_str)
            if pd.isna(ts):
                continue
            combined.setdefault(ts, {})
            if isinstance(entry, dict):
                for field, val in entry.items():
                    combined[ts][f"ned_{etype}_{field}"] = safe_float(val)
            else:
                combined[ts][f"ned_{etype}"] = safe_float(entry)

    rows = []
    for ts, vals in combined.items():
        rows.append({"timestamp": ts, **meta(month), **vals})
    return rows

# grid imbalance: imbalance_price, balance_delta, direction

def parse_grid_imbalance(month, filepath):
    d = safe_load_json(filepath)
    if not d:
        return []

    data_section = d.get("data", {})
    if not isinstance(data_section, dict):
        return []

    series_map = {
        "imbalance_price": "grid_imbalance_price_eur_mwh",
        "balance_delta":   "grid_balance_delta_mw",
        "direction":       "grid_direction",
    }

    combined = {}
    for series_key, col_name in series_map.items():
        series = data_section.get(series_key, {})
        if not isinstance(series, dict):
            continue
        for ts_str, val in series.items():
            ts = to_utc(ts_str)
            if pd.isna(ts):
                continue
            combined.setdefault(ts, {})[col_name] = safe_float(val)

    rows = []
    for ts, vals in combined.items():
        rows.append({"timestamp": ts, **meta(month), **vals})
    return rows

# gas flows: daily data, the merger forward-fills it to hourly later

def parse_gas_flows(month, filepath):
    d = safe_load_json(filepath)
    if not d:
        return []

    data_dict = d.get("data", {})
    if not isinstance(data_dict, dict) or len(data_dict) == 0:
        return []

    rows = []
    for ts_str, entry in data_dict.items():
        ts = to_utc(ts_str)
        if pd.isna(ts) or not isinstance(entry, dict):
            continue
        rows.append({
            "timestamp":             ts,
            **meta(month),
            "gas_entry_total_gwh":   safe_float(entry.get("entry_total_gwh")),
            "gas_exit_total_gwh":    safe_float(entry.get("exit_total_gwh")),
            "gas_net_flow_gwh":      safe_float(entry.get("net_flow_gwh")),
        })
    return rows

# run every parser. gas_storage is skipped (its data is empty).

PARSER_CONFIG = [
    ("energy_price_forecast",           parse_energy,         "parsed_energy"),
    ("weather_forecast_multi_location", parse_weather,        "parsed_weather"),
    ("wind_forecast",                   parse_wind,           "parsed_wind"),
    ("solar_forecast",                  parse_solar,          "parsed_solar"),
    ("demand_weather_forecast",         parse_demand_weather, "parsed_demand_weather"),
    ("calendar_features",               parse_calendar,       "parsed_calendar"),
    ("cross_border_flows",              parse_cross_border,   "parsed_cross_border"),
    ("load_forecast",                   parse_load,           "parsed_load"),
    ("generation_forecast",             parse_generation,     "parsed_generation"),
    ("market_proxies",                  parse_market_proxies, "parsed_market_proxies"),
    ("ned_production",                  parse_ned,            "parsed_ned_production"),
    ("grid_imbalance",                  parse_grid_imbalance, "parsed_grid_imbalance"),
    ("gas_flows",                       parse_gas_flows,      "parsed_gas_flows"),
]

print("=" * 70)
print("SCRIPT 02 - PARSER")
print("=" * 70)
print(f"Data folder  : {DATA_DIR}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Months       : {TARGET_MONTHS}")
print(f"Forecast months: {sorted(FORECAST_MONTHS)}")
print("=" * 70)

total_rows_all = 0

for keyword, parser_fn, output_name in PARSER_CONFIG:
    all_rows   = []
    file_count = 0
    skip_count = 0

    print(f"\nParsing [{output_name}]...")

    for month in TARGET_MONTHS:
        files = get_files(month, keyword)
        for filepath in files:
            rows = parser_fn(month, filepath)
            if rows:
                all_rows.extend(rows)
                file_count += 1
            else:
                skip_count += 1

    if not all_rows:
        print(f"  -> no data found, skipping")
        continue

    df = pd.DataFrame(all_rows)

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    out_path = os.path.join(OUTPUT_DIR, f"{output_name}.parquet")
    df.to_parquet(out_path, index=False)
    total_rows_all += len(df)

    print(f"  -> files parsed: {file_count}  |  skipped (empty/encrypted): {skip_count}")
    print(f"  -> rows saved  : {len(df):,}")
    print(f"  -> columns     : {len(df.columns)}  {list(df.columns)[:6]}{'...' if len(df.columns)>6 else ''}")
    print(f"  -> date range  : {df['timestamp'].min()} -> {df['timestamp'].max()}")

print("\n" + "=" * 70)
print(f"DONE - {total_rows_all:,} total rows across all parquet files. next run script 03.")
print("=" * 70)
