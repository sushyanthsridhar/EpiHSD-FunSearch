"""
build_input.py
--------------
Merges all feature sources into a single input.csv for modelling.

Current sources
  1. cases      : demographics/timeseries_all_provinces.csv
  2. temperature : weather/temperature/weekly_weather_by_province.csv
  3. rainfall   : weather/rainfall/weekly_rainfall_by_province.csv
  4. humidity   : weather/humidity/weekly_humidity_by_province.csv
  5. population : population/province_population.csv  (annual, merged on province+year)

Output
  data_processing/input.csv
  columns: province, year, month, week, total_cases,
           avg_temp, max_temp, min_temp,
           avg_rainfall, max_rainfall, min_rainfall,
           avg_humidity, max_humidity, min_humidity,
           population
"""

import pandas as pd
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
CASES      = os.path.join(BASE, "demographics",  "timeseries_all_provinces.csv")
WEATHER    = os.path.join(BASE, "weather", "temperature", "weekly_weather_by_province.csv")
RAINFALL   = os.path.join(BASE, "weather", "rainfall",    "weekly_rainfall_by_province.csv")
HUMIDITY   = os.path.join(BASE, "weather", "humidity",    "weekly_humidity_by_province.csv")
POPULATION = os.path.join(BASE, "population", "province_population.csv")
OUTPUT     = os.path.join(BASE, "input.csv")

WEEKLY_KEYS = ["province", "year", "week"]    # for weather merges
ANNUAL_KEYS = ["province", "year"]            # for population merge (annual data)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_and_check(path, label):
    df = pd.read_csv(path)
    df = df.sort_values([c for c in WEEKLY_KEYS if c in df.columns]).reset_index(drop=True)
    print(f"  {label:14s}: {len(df):,} rows | {df['province'].nunique()} provinces | cols: {list(df.columns)}")
    return df

def merge_weekly(base, feature, label, new_cols):
    merged = base.merge(feature[WEEKLY_KEYS + new_cols], on=WEEKLY_KEYS, how="left")
    n_miss = merged[new_cols].isnull().sum().sum()
    status = "OK " if n_miss == 0 else f"WARNING: {n_miss} unmatched"
    print(f"  {status}  [{label}]")
    return merged

def merge_annual(base, feature, label, new_cols):
    """Population is annual — merge on province+year, broadcasts to all 52 weeks."""
    merged = base.merge(feature[ANNUAL_KEYS + new_cols], on=ANNUAL_KEYS, how="left")
    n_miss = merged[new_cols].isnull().sum().sum()
    status = "OK " if n_miss == 0 else f"WARNING: {n_miss} unmatched"
    print(f"  {status}  [{label}] (annual -> broadcast to all weeks)")
    return merged

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading sources ...")
cases      = load_and_check(CASES,      "cases")
weather    = load_and_check(WEATHER,    "temperature")
rainfall   = load_and_check(RAINFALL,   "rainfall")
humidity   = load_and_check(HUMIDITY,   "humidity")
population = load_and_check(POPULATION, "population")

# ── 2. Merge ──────────────────────────────────────────────────────────────────
print("\nMerging ...")
df = merge_weekly(cases, weather,    "temperature", ["avg_temp",     "max_temp",     "min_temp"])
df = merge_weekly(df,    rainfall,   "rainfall",    ["avg_rainfall", "max_rainfall", "min_rainfall"])
df = merge_weekly(df,    humidity,   "humidity",    ["avg_humidity", "max_humidity", "min_humidity"])
df = merge_annual(df,    population, "population",  ["population"])

# ── 3. Column order ───────────────────────────────────────────────────────────
col_order = [
    "province", "year", "month", "week",
    "total_cases",
    "avg_temp",     "max_temp",     "min_temp",
    "avg_rainfall", "max_rainfall", "min_rainfall",
    "avg_humidity", "max_humidity", "min_humidity",
    "population",
]
df = df[col_order]

# ── 4. Save ───────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT, index=False)

print(f"\nSaved  ->  {OUTPUT}")
print(f"Shape  :   {df.shape}  ({df['province'].nunique()} provinces x {df['year'].nunique()} years x 52 weeks)")
print(f"\nMissing values per column:")
print(df.isnull().sum().to_string())
print(f"\nFirst 3 rows:")
print(df.head(3).to_string(index=False))
print(f"\nFeature summary:")
feat_cols = [c for c in col_order if c not in ["province","year","month","week"]]
print(df[feat_cols].describe().round(2).to_string())
