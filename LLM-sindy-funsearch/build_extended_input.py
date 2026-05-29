"""
build_extended_input.py
────────────────────────────────────────────────────────────────
Final assembly script. Merges input.csv with all newly fetched
feature files into one extended_input.csv.

Run this LAST, after all scripts in the data/ subfolders.

Missing feature files are handled gracefully. If a file does
not exist yet (e.g. GEE not set up), those columns are simply
filled with NaN and a warning is printed.

Output: extended_input.csv
────────────────────────────────────────────────────────────────
"""

import pandas as pd
import os

HERE       = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV  = os.path.join(HERE, "input.csv")
OUTPUT_CSV = os.path.join(HERE, "extended_input.csv")

# ── Feature file registry ─────────────────────────────────────
# Each entry: (filepath, merge_keys, description)
FEATURE_FILES = [
    (
        os.path.join(HERE, "data/1_derived/derived_features.csv"),
        ["province", "year", "week"],
        "Derived climate features (diurnal range, dry/wet streaks, etc.)"
    ),
    (
        os.path.join(HERE, "data/2_spatial/neighbor_features.csv"),
        ["province", "year", "week"],
        "Spatial neighbor incidence features"
    ),
    (
        os.path.join(HERE, "data/3_climate_indices/enso_indices.csv"),
        ["year", "week"],
        "ENSO climate indices (Nino3.4, ONI, SOI)"
    ),
    (
        os.path.join(HERE, "data/3_climate_indices/sst_indices.csv"),
        ["year", "week"],
        "Caribbean SST indices (TNA, AMM)"
    ),
    (
        os.path.join(HERE, "data/4_nasa_power/nasa_power_features.csv"),
        ["province", "year", "week"],
        "NASA POWER (solar radiation, wind speed, dew point)"
    ),
    (
        os.path.join(HERE, "data/5_remote_sensing/ndvi_lst_features.csv"),
        ["province", "year", "week"],
        "Remote sensing NDVI and land surface temperature"
    ),
]

# ── Load base ─────────────────────────────────────────────────
print("Loading input.csv ...")
df = pd.read_csv(INPUT_CSV)
print(f"  Base: {len(df):,} rows, {len(df.columns)} columns.")

# ── Merge each feature file ────────────────────────────────────
for filepath, keys, description in FEATURE_FILES:
    if not os.path.exists(filepath):
        print(f"  SKIP (not found): {description}")
        continue

    feat_df = pd.read_csv(filepath)

    # Drop columns already in df to avoid duplicates
    overlap = [c for c in feat_df.columns if c in df.columns and c not in keys]
    if overlap:
        feat_df = feat_df.drop(columns=overlap)

    # Deduplicate on merge keys by averaging — prevents row doubling
    # if any source file accidentally has duplicate key combinations
    feat_df = feat_df.groupby(keys).mean().reset_index()

    # Drop columns that are 100% NaN (e.g. GEE placeholders)
    all_nan = [c for c in feat_df.columns if c not in keys and feat_df[c].isna().all()]
    if all_nan:
        feat_df = feat_df.drop(columns=all_nan)
        print(f"  NOTE: dropped 100% NaN columns {all_nan} from {description}")

    if len(feat_df.columns) == len(keys):
        print(f"  SKIP (all columns NaN): {description}")
        continue

    before = len(df.columns)
    df = df.merge(feat_df, on=keys, how="left")
    added = len(df.columns) - before

    print(f"  MERGED: {description}  (+{added} columns)")

# ── Summary ───────────────────────────────────────────────────
print(f"\nFinal dataset: {len(df):,} rows, {len(df.columns)} columns.")
print(f"Columns: {list(df.columns)}")

# Report NaN rates for new columns
base_cols = pd.read_csv(INPUT_CSV, nrows=0).columns.tolist()
new_cols  = [c for c in df.columns if c not in base_cols]
if new_cols:
    print("\nNaN rates for new features:")
    for col in new_cols:
        nan_pct = df[col].isna().mean() * 100
        if nan_pct > 0:
            print(f"  {col}: {nan_pct:.1f}% missing")

# ── Save ──────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved extended_input.csv  ({len(df):,} rows, {len(df.columns)} columns)")
print("This is your new master dataset for the autoencoder pipeline.")
