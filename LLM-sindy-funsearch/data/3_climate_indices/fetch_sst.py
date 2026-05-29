"""
fetch_sst.py
────────────────────────────────────────────────────────────────
Downloads Caribbean and Tropical North Atlantic SST indices
from NOAA PSL. No account needed.

Indices fetched:
  - tna_sst_anom  : Tropical North Atlantic SST anomaly
                    (most directly relevant to DR and Caribbean dengue)
  - caribbean_sst_anom : Caribbean regional SST proxy (AMM index)

Output: sst_indices.csv
────────────────────────────────────────────────────────────────
"""

import requests
import pandas as pd
import numpy as np
import os

HERE       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(HERE, "sst_indices.csv")

YEARS = list(range(2015, 2024))

def week_to_month(week):
    return min(int((week - 1) / 4.33) + 1, 12)

def fetch_psl_index(url, col_name, missing_val=-99.99):
    """
    Parse a NOAA PSL climate index file.
    Format: year followed by 12 monthly values per row.
    """
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        records = []
        for line in lines:
            parts = line.split()
            if len(parts) == 13 and parts[0].isdigit():
                yr = int(parts[0])
                if yr in YEARS:
                    for mon_idx, val_str in enumerate(parts[1:], start=1):
                        try:
                            val = float(val_str)
                            if abs(val - missing_val) > 1.0:
                                records.append({"year": yr, "month": mon_idx, col_name: val})
                        except ValueError:
                            continue
        df = pd.DataFrame(records)
        print(f"  Got {len(df)} monthly records for {col_name}.")
        return df
    except Exception as e:
        print(f"  Warning: Could not fetch {col_name} ({e}).")
        return pd.DataFrame(columns=["year", "month", col_name])

# ── Tropical North Atlantic SST anomaly ───────────────────────
print("Fetching Tropical North Atlantic SST index ...")
tna_df = fetch_psl_index(
    "https://psl.noaa.gov/data/correlation/tna.data",
    "tna_sst_anom"
)

# ── Atlantic Meridional Mode (AMM) as Caribbean SST proxy ─────
print("Fetching AMM index (Caribbean SST proxy) ...")
amm_df = fetch_psl_index(
    "https://psl.noaa.gov/data/correlation/ammsst.data",
    "caribbean_sst_anom"
)

# ── Build weekly grid ──────────────────────────────────────────
print("Interpolating to weekly resolution ...")
rows = []
for yr in YEARS:
    for wk in range(1, 53):
        rows.append({"year": yr, "week": wk, "month": week_to_month(wk)})
weekly = pd.DataFrame(rows)

if not tna_df.empty:
    weekly = weekly.merge(tna_df, on=["year", "month"], how="left")
else:
    weekly["tna_sst_anom"] = 0.0

if not amm_df.empty:
    weekly = weekly.merge(amm_df, on=["year", "month"], how="left")
else:
    weekly["caribbean_sst_anom"] = 0.0

weekly = weekly.sort_values(["year", "week"])
for col in ["tna_sst_anom", "caribbean_sst_anom"]:
    weekly[col] = weekly[col].ffill().fillna(0.0)

out = weekly[["year", "week", "tna_sst_anom", "caribbean_sst_anom"]]
out.to_csv(OUTPUT_CSV, index=False)
print(f"Done. Saved {len(out)} rows to {OUTPUT_CSV}")
