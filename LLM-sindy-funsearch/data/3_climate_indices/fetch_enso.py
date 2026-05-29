"""
fetch_enso.py
────────────────────────────────────────────────────────────────
Downloads ENSO climate indices from NOAA and interpolates to
weekly resolution matching input.csv (2015 to 2023).

No account or API key needed. Direct file download from NOAA.

Indices fetched:
  - nino34_anom   : Nino 3.4 SST anomaly (core ENSO index)
  - oni_index     : Oceanic Nino Index (3-month running mean)
  - soi_index     : Southern Oscillation Index (atmospheric ENSO)

Output: enso_indices.csv
────────────────────────────────────────────────────────────────
"""

import requests
import pandas as pd
import numpy as np
import io
import os

HERE       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(HERE, "enso_indices.csv")

YEARS = list(range(2015, 2024))

# ── Helper: week number to approximate month ───────────────────
def week_to_month(week):
    return min(int((week - 1) / 4.33) + 1, 12)

# ── 1. Nino 3.4 anomaly ────────────────────────────────────────
print("Fetching Nino 3.4 index from NOAA ...")
url_nino = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
try:
    r = requests.get(url_nino, timeout=30)
    r.raise_for_status()
    lines = r.text.strip().split("\n")
    # Format: YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM
    records = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 8:
            try:
                yr  = int(parts[0])
                mon = int(parts[1])
                nino34 = float(parts[7])  # NINO3.4 anomaly column
                if yr in YEARS and nino34 != -99.9:
                    records.append({"year": yr, "month": mon, "nino34_anom": nino34})
            except ValueError:
                continue
    nino_df = pd.DataFrame(records)
    print(f"  Got {len(nino_df)} monthly Nino3.4 records.")
except Exception as e:
    print(f"  Warning: Could not fetch Nino3.4 ({e}). Using zeros.")
    nino_df = pd.DataFrame(columns=["year", "month", "nino34_anom"])

# ── 2. ONI index ───────────────────────────────────────────────
print("Fetching ONI index from NOAA ...")
url_oni = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
season_map = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12
}
try:
    r = requests.get(url_oni, timeout=30)
    r.raise_for_status()
    lines = r.text.strip().split("\n")
    records = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            yr  = int(parts[0])
            szn = parts[1]
            val = float(parts[2])
            mon = season_map.get(szn, None)
            if yr in YEARS and mon and val != -99.9:
                records.append({"year": yr, "month": mon, "oni_index": val})
    oni_df = pd.DataFrame(records)
    print(f"  Got {len(oni_df)} monthly ONI records.")
except Exception as e:
    print(f"  Warning: Could not fetch ONI ({e}). Using zeros.")
    oni_df = pd.DataFrame(columns=["year", "month", "oni_index"])

# ── 3. SOI index ───────────────────────────────────────────────
print("Fetching SOI index from NOAA ...")
url_soi = "https://www.cpc.ncep.noaa.gov/data/indices/soi"
try:
    r = requests.get(url_soi, timeout=30)
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
                        if val != -999.9:
                            records.append({"year": yr, "month": mon_idx, "soi_index": val})
                    except ValueError:
                        continue
    soi_df = pd.DataFrame(records)
    print(f"  Got {len(soi_df)} monthly SOI records.")
except Exception as e:
    print(f"  Warning: Could not fetch SOI ({e}). Using zeros.")
    soi_df = pd.DataFrame(columns=["year", "month", "soi_index"])

# ── Build weekly grid and merge ────────────────────────────────
print("Interpolating to weekly resolution ...")
rows = []
for yr in YEARS:
    for wk in range(1, 53):
        mon = week_to_month(wk)
        rows.append({"year": yr, "week": wk, "month": mon})
weekly = pd.DataFrame(rows)

# Merge monthly indices onto weekly grid by year+month
if not nino_df.empty:
    weekly = weekly.merge(nino_df[["year", "month", "nino34_anom"]],
                          on=["year", "month"], how="left")
else:
    weekly["nino34_anom"] = 0.0

if not oni_df.empty:
    weekly = weekly.merge(oni_df[["year", "month", "oni_index"]],
                          on=["year", "month"], how="left")
else:
    weekly["oni_index"] = 0.0

if not soi_df.empty:
    weekly = weekly.merge(soi_df[["year", "month", "soi_index"]],
                          on=["year", "month"], how="left")
else:
    weekly["soi_index"] = 0.0

# Forward fill any gaps then fill remaining with 0
weekly = weekly.sort_values(["year", "week"])
for col in ["nino34_anom", "oni_index", "soi_index"]:
    weekly[col] = weekly[col].ffill().fillna(0.0)

out = weekly[["year", "week", "nino34_anom", "oni_index", "soi_index"]]
out.to_csv(OUTPUT_CSV, index=False)
print(f"Done. Saved {len(out)} rows to {OUTPUT_CSV}")
print("These indices are global so they will be the same value for all provinces.")
print("build_extended_input.py will broadcast them across all provinces automatically.")
