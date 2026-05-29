"""
fetch_nasa_power.py
────────────────────────────────────────────────────────────────
Fetches additional climate variables from the NASA POWER API
for each of the 32 DR provinces.

No account or API key needed. Completely free.

Variables fetched per province centroid:
  - solar_radiation   : surface solar radiation (kWh/m2/day)
  - wind_speed        : wind speed at 2m height (m/s)
  - dew_point_temp    : dew point temperature at 2m (Celsius)

Daily values are aggregated to weekly means to match input.csv.

NOTE: This script fetches data for 32 provinces one at a time.
      It takes approximately 10 to 20 minutes to complete.
      A progress counter shows which province is being fetched.
      If it fails midway, re-run it. Already-fetched provinces
      are skipped automatically.

Output: nasa_power_features.csv
────────────────────────────────────────────────────────────────
"""

import requests
import pandas as pd
import numpy as np
import time
import os
import json

HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.dirname(HERE)
CENTROIDS   = os.path.join(DATA_DIR, "province_centroids.csv")
OUTPUT_CSV  = os.path.join(HERE, "nasa_power_features.csv")
CACHE_DIR   = os.path.join(HERE, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

START = "20150101"
END   = "20231231"

NASA_PARAMS = "ALLSKY_SFC_SW_DWN,WS2M,T2MDEW"
# ALLSKY_SFC_SW_DWN : All sky surface shortwave downward irradiance (solar radiation)
# WS2M              : Wind speed at 2 meters
# T2MDEW            : Dew point temperature at 2 meters

API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# ── Load centroids ─────────────────────────────────────────────
centroids = pd.read_csv(CENTROIDS)
centroids["prov_key"] = centroids["province_id"].astype(str).str.zfill(2)

all_results = []

for i, row in centroids.iterrows():
    prov_key  = row["prov_key"]
    prov_name = row["province_name"]
    lat       = row["latitude"]
    lon       = row["longitude"]

    cache_file = os.path.join(CACHE_DIR, f"{prov_key}.json")

    print(f"[{i+1:2d}/32] {prov_name} ...", end=" ", flush=True)

    # Load from cache if already fetched
    if os.path.exists(cache_file):
        print("(cached)")
        with open(cache_file) as f:
            data = json.load(f)
    else:
        # Fetch from NASA POWER
        params = {
            "parameters": NASA_PARAMS,
            "community":  "RE",
            "longitude":  lon,
            "latitude":   lat,
            "start":      START,
            "end":        END,
            "format":     "JSON",
        }
        try:
            r = requests.get(API_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            with open(cache_file, "w") as f:
                json.dump(data, f)
            print("OK")
            time.sleep(1.5)  # be polite to the API
        except Exception as e:
            print(f"FAILED ({e})")
            continue

    # ── Parse daily data ──────────────────────────────────────
    try:
        props = data["properties"]["parameter"]
        solar = props.get("ALLSKY_SFC_SW_DWN", {})
        wind  = props.get("WS2M", {})
        dew   = props.get("T2MDEW", {})

        rows = []
        for date_str, sol_val in solar.items():
            yr  = int(date_str[:4])
            mon = int(date_str[4:6])
            day = int(date_str[6:8])

            # ISO week number
            import datetime
            wk = datetime.date(yr, mon, day).isocalendar()[1]

            # Clamp week 53 to 52
            wk = min(wk, 52)

            rows.append({
                "year":           yr,
                "week":           wk,
                "solar_radiation": sol_val if sol_val != -999.0 else np.nan,
                "wind_speed":     wind.get(date_str, np.nan) if wind.get(date_str, -999.0) != -999.0 else np.nan,
                "dew_point_temp": dew.get(date_str, np.nan)  if dew.get(date_str,  -999.0) != -999.0 else np.nan,
            })

        daily_df = pd.DataFrame(rows)

        # Aggregate daily to weekly mean
        weekly_df = daily_df.groupby(["year", "week"]).mean().reset_index()
        weekly_df["province_key"] = prov_key
        weekly_df["province_name"] = prov_name

        # Map province_key back to full province label used in input.csv
        weekly_df["province"] = prov_key + " " + prov_name

        all_results.append(weekly_df)

    except Exception as e:
        print(f"  Parse error for {prov_name}: {e}")
        continue

# ── Combine and save ───────────────────────────────────────────
if all_results:
    out = pd.concat(all_results, ignore_index=True)
    out = out[["province", "year", "week",
               "solar_radiation", "wind_speed", "dew_point_temp"]]
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone. Saved {len(out):,} rows to {OUTPUT_CSV}")
else:
    print("\nNo data fetched. Check your internet connection and try again.")
