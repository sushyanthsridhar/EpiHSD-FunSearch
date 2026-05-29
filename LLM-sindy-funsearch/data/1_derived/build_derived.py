"""
build_derived.py
────────────────────────────────────────────────────────────────
Computes new features entirely from columns already in input.csv.
No internet connection required. Run this first.

Output: derived_features.csv
Columns added:
  - diurnal_temp_range       : max_temp minus min_temp (daily heat stress on mosquitoes)
  - consecutive_dry_days     : rolling weeks of rainfall below 1mm (drought stress)
  - consecutive_wet_days     : rolling weeks of rainfall above 10mm (breeding site creation)
  - extreme_rainfall_binary  : 1 if that week's rainfall exceeds province 95th percentile
  - cooling_degree_days      : max(0, avg_temp - 18) proxy for heat accumulation
  - rainy_season_phase       : 1 if month is May through November (DR wet season)
  - school_calendar_active   : 1 during DR school term weeks (affects contact rates)
────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(os.path.dirname(HERE))
INPUT_CSV  = os.path.join(ROOT, "input.csv")
OUTPUT_CSV = os.path.join(HERE, "derived_features.csv")

# ── Load ───────────────────────────────────────────────────────
print("Loading input.csv ...")
df = pd.read_csv(INPUT_CSV)
print(f"  {len(df):,} rows loaded.")

# Sort so rolling windows work correctly within each province
df = df.sort_values(["province", "year", "week"]).reset_index(drop=True)

results = []

for province, grp in df.groupby("province"):
    grp = grp.copy().reset_index(drop=True)

    # 1. Diurnal temperature range
    grp["diurnal_temp_range"] = grp["max_temp"] - grp["min_temp"]

    # 2. Consecutive dry weeks (rolling count, resets on wet week)
    dry = (grp["avg_rainfall"] < 1.0).astype(int)
    consec_dry = []
    count = 0
    for val in dry:
        count = count + 1 if val == 1 else 0
        consec_dry.append(count)
    grp["consecutive_dry_weeks"] = consec_dry

    # 3. Consecutive wet weeks
    wet = (grp["avg_rainfall"] >= 10.0).astype(int)
    consec_wet = []
    count = 0
    for val in wet:
        count = count + 1 if val == 1 else 0
        consec_wet.append(count)
    grp["consecutive_wet_weeks"] = consec_wet

    # 4. Extreme rainfall binary (above province 95th percentile)
    p95 = grp["avg_rainfall"].quantile(0.95)
    grp["extreme_rainfall_binary"] = (grp["avg_rainfall"] > p95).astype(int)

    # 5. Cooling degree days proxy (heat accumulation above 18 Celsius base)
    grp["cooling_degree_days"] = (grp["avg_temp"] - 18.0).clip(lower=0)

    # 6. Rainy season phase (DR wet season: May to November, months 5 to 11)
    grp["rainy_season_phase"] = grp["month"].isin([5, 6, 7, 8, 9, 10, 11]).astype(int)

    # 7. School calendar active
    # DR school year roughly: week 5 to week 26 (Feb to Jun) and week 35 to week 51 (Sep to Dec)
    grp["school_calendar_active"] = (
        grp["week"].between(5, 26) | grp["week"].between(35, 51)
    ).astype(int)

    results.append(grp)

# ── Combine and save ───────────────────────────────────────────
out = pd.concat(results, ignore_index=True)

# Keep only the key columns needed for merging plus the new features
new_cols = [
    "province", "year", "week",
    "diurnal_temp_range",
    "consecutive_dry_weeks",
    "consecutive_wet_weeks",
    "extreme_rainfall_binary",
    "cooling_degree_days",
    "rainy_season_phase",
    "school_calendar_active",
]
out[new_cols].to_csv(OUTPUT_CSV, index=False)

print(f"Done. Saved {len(out):,} rows to {OUTPUT_CSV}")
print(f"New features: {new_cols[3:]}")
