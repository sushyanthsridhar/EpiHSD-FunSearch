"""
fetch_ndvi_lst.py
────────────────────────────────────────────────────────────────
Fetches NDVI and Land Surface Temperature (LST) for each
DR province from NASA MODIS via Google Earth Engine.

NDVI (vegetation greenness) correlates with mosquito breeding habitat.
LST day/night captures thermal conditions for vector activity
more accurately than air temperature.

── SETUP REQUIRED (one time only) ────────────────────────────
1. Create a free Google account if you don't have one.
2. Sign up for Google Earth Engine at: https://earthengine.google.com
   (free for research, approval takes minutes to hours)
3. Install the Python package:
      pip install earthengine-api
4. Authenticate once in your terminal:
      earthengine authenticate
   A browser window will open. Log in and paste the code back.
5. Then run this script normally:
      python fetch_ndvi_lst.py

── WHAT THIS SCRIPT DOES ─────────────────────────────────────
- Uses MODIS MOD13Q1 (16-day NDVI composite, 250m resolution)
- Uses MODIS MOD11A2 (8-day LST composite, 1km resolution)
- Extracts mean values within a 20km buffer around each
  province centroid (approximates province-level average)
- Aggregates to monthly then maps to weekly to match input.csv

Output: ndvi_lst_features.csv
Columns: province, year, week, ndvi_mean, lst_day_mean, lst_night_mean
────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import numpy as np

HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.dirname(HERE)
CENTROIDS   = os.path.join(DATA_DIR, "province_centroids.csv")
OUTPUT_CSV  = os.path.join(HERE, "ndvi_lst_features.csv")

YEARS = list(range(2015, 2024))

def week_to_month(week):
    return min(int((week - 1) / 4.33) + 1, 12)

# ── Load centroids ─────────────────────────────────────────────
centroids = pd.read_csv(CENTROIDS)
centroids["prov_key"] = centroids["province_id"].astype(str).str.zfill(2)

# ── Try to import Earth Engine ─────────────────────────────────
try:
    import ee
    ee.Initialize()
    print("Google Earth Engine initialized.")
    GEE_AVAILABLE = True
except Exception as e:
    print(f"Google Earth Engine not available: {e}")
    print("Please complete the setup steps in the script header.")
    GEE_AVAILABLE = False

if not GEE_AVAILABLE:
    print("\nCreating placeholder output with NaN values.")
    print("Run this script again after completing GEE setup.")
    rows = []
    for yr in YEARS:
        for wk in range(1, 53):
            for _, row in centroids.iterrows():
                rows.append({
                    "province":      row["prov_key"] + " " + row["province_name"],
                    "year":          yr,
                    "week":          wk,
                    "ndvi_mean":     np.nan,
                    "lst_day_mean":  np.nan,
                    "lst_night_mean": np.nan,
                })
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"Placeholder saved to {OUTPUT_CSV}")
    exit()

# ── Fetch via GEE ─────────────────────────────────────────────
all_results = []

ndvi_collection = ee.ImageCollection("MODIS/006/MOD13Q1").select("NDVI")
lst_collection  = ee.ImageCollection("MODIS/006/MOD11A2").select(["LST_Day_1km", "LST_Night_1km"])

for i, row in centroids.iterrows():
    prov_key  = row["prov_key"]
    prov_name = row["province_name"]
    lat       = row["latitude"]
    lon       = row["longitude"]

    print(f"[{i+1:2d}/32] Fetching {prov_name} ...", flush=True)

    # 20km buffer around centroid as province proxy
    point  = ee.Geometry.Point([lon, lat])
    region = point.buffer(20000)

    monthly_records = []

    for yr in YEARS:
        for mon in range(1, 13):
            start = f"{yr}-{mon:02d}-01"
            end   = f"{yr}-{mon:02d}-28" if mon == 2 else f"{yr}-{mon:02d}-30"

            # NDVI mean for this month
            try:
                ndvi_img = ndvi_collection.filterDate(start, end).mean()
                ndvi_val = ndvi_img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region,
                    scale=250,
                ).getInfo().get("NDVI", None)
                # MODIS NDVI scale factor is 0.0001
                ndvi_val = ndvi_val * 0.0001 if ndvi_val is not None else np.nan
            except Exception:
                ndvi_val = np.nan

            # LST mean for this month
            try:
                lst_img     = lst_collection.filterDate(start, end).mean()
                lst_result  = lst_img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region,
                    scale=1000,
                ).getInfo()
                lst_day   = lst_result.get("LST_Day_1km",   None)
                lst_night = lst_result.get("LST_Night_1km", None)
                # MODIS LST scale factor: 0.02, then subtract 273.15 for Celsius
                lst_day   = lst_day   * 0.02 - 273.15 if lst_day   is not None else np.nan
                lst_night = lst_night * 0.02 - 273.15 if lst_night is not None else np.nan
            except Exception:
                lst_day   = np.nan
                lst_night = np.nan

            monthly_records.append({
                "year":          yr,
                "month":         mon,
                "ndvi_mean":     ndvi_val,
                "lst_day_mean":  lst_day,
                "lst_night_mean": lst_night,
            })

    monthly_df = pd.DataFrame(monthly_records)

    # Expand monthly to weekly
    for yr in YEARS:
        for wk in range(1, 53):
            mon = week_to_month(wk)
            mrow = monthly_df[(monthly_df["year"] == yr) & (monthly_df["month"] == mon)]
            if not mrow.empty:
                all_results.append({
                    "province":      prov_key + " " + prov_name,
                    "year":          yr,
                    "week":          wk,
                    "ndvi_mean":     mrow["ndvi_mean"].values[0],
                    "lst_day_mean":  mrow["lst_day_mean"].values[0],
                    "lst_night_mean": mrow["lst_night_mean"].values[0],
                })

# ── Save ───────────────────────────────────────────────────────
out = pd.DataFrame(all_results)
out.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone. Saved {len(out):,} rows to {OUTPUT_CSV}")
