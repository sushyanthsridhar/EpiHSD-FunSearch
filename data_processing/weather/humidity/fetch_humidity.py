"""
fetch_humidity.py
-----------------
Fetches weekly relative humidity data (avg, max, min) for each Dominican Republic
province from the Open-Meteo Historical Weather API (ERA5 reanalysis).

Since Open-Meteo returns humidity as hourly data, this script:
  1. Fetches hourly relative_humidity_2m
  2. Aggregates hourly -> daily (mean, max, min)
  3. Aggregates daily  -> weekly ISO week (mean of means, max of maxes, min of mins)

Output: weekly_humidity_by_province.csv
  columns: province, year, week, avg_humidity, max_humidity, min_humidity
  units  : % relative humidity

Why humidity matters for dengue (SEIR context):
  - High humidity extends adult Aedes aegypti lifespan (lower mortality rate μ)
  - Affects biting frequency and host-seeking behaviour
  - Combined with temperature, determines mosquito vectorial capacity

Merge into input.csv (after running build_input.py):
    import pandas as pd
    df  = pd.read_csv("../input.csv")
    hum = pd.read_csv("weekly_humidity_by_province.csv")
    df  = df.merge(hum, on=["province", "year", "week"], how="left")
    df.to_csv("../input.csv", index=False)

Re-run safe: already-fetched provinces are skipped automatically.
"""

import time
import requests
import pandas as pd
import os

# ── Province centroids (latitude, longitude) ──────────────────────────────────
PROVINCE_COORDS = {
    "01 Distrito Nacional"     : (18.4861, -69.9312),
    "02 Azua"                  : (18.4521, -70.7349),
    "03 Baoruco"               : (18.4887, -71.4205),
    "04 Barahona"              : (18.2134, -71.0998),
    "05 Dajabón"               : (19.5497, -71.7082),
    "06 Duarte"                : (19.2100, -70.0299),
    "07 Elías Piña"            : (18.8763, -71.7070),
    "08 El Seibo"              : (18.7665, -69.0397),
    "09 Espaillat"             : (19.6247, -70.2703),
    "10 Independencia"         : (18.5075, -71.8466),
    "11 La Altagracia"         : (18.5736, -68.6230),
    "12 La Romana"             : (18.4273, -68.9728),
    "13 La Vega"               : (19.2212, -70.5290),
    "14 María Trinidad Sánchez": (19.3667, -69.8522),
    "15 Monte Cristi"          : (19.8607, -71.6526),
    "16 Pedernales"            : (18.0380, -71.7437),
    "17 Peravia"               : (18.2798, -70.3298),
    "18 Puerto Plata"          : (19.7935, -70.6906),
    "19 Hermanas Mirabal"      : (19.3780, -70.4148),
    "20 Samaná"                : (19.2061, -69.3364),
    "21 San Cristóbal"         : (18.4175, -70.1069),
    "22 San Juan"              : (18.8061, -71.2286),
    "23 San Pedro de Macorís"  : (18.4596, -69.3055),
    "24 Sánchez Ramírez"       : (19.0517, -70.1497),
    "25 Santiago"              : (19.4517, -70.6970),
    "26 Santiago Rodríguez"    : (19.4762, -71.3400),
    "27 Valverde"              : (19.5895, -70.9661),
    "28 Monseñor Nouel"        : (18.9225, -70.3843),
    "29 Monte Plata"           : (18.8068, -69.7809),
    "30 Hato Mayor"            : (18.7644, -69.2559),
    "31 San José de Ocoa"      : (18.5438, -70.5027),
    "32 Santo Domingo"         : (18.5001, -69.9884),
}

# ── Configuration ─────────────────────────────────────────────────────────────
START_DATE  = "2015-01-01"
END_DATE    = "2023-12-31"
API_URL     = "https://archive-api.open-meteo.com/v1/archive"
OUTPUT_FILE = "weekly_humidity_by_province.csv"
SLEEP_SEC   = 1.5    # slightly longer — hourly payload is larger
MAX_RETRIES = 3
RETRY_WAIT  = 5.0


def fetch_hourly_humidity(lat: float, lon: float) -> pd.DataFrame:
    """Fetch hourly relative_humidity_2m from Open-Meteo archive."""
    params = {
        "latitude"  : lat,
        "longitude" : lon,
        "start_date": START_DATE,
        "end_date"  : END_DATE,
        "hourly"    : "relative_humidity_2m",
        "timezone"  : "America/Santo_Domingo",
    }
    resp = requests.get(API_URL, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame({
        "datetime": pd.to_datetime(data["time"]),
        "rh"      : pd.to_numeric(data["relative_humidity_2m"], errors="coerce"),
    })
    df["rh"] = df["rh"].ffill().bfill()
    return df


def hourly_to_weekly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly RH -> daily -> weekly ISO week."""
    hourly = hourly.copy()
    hourly["date"] = hourly["datetime"].dt.date

    # Step 1: hourly -> daily
    daily = (
        hourly.groupby("date")["rh"]
        .agg(rh_mean="mean", rh_max="max", rh_min="min")
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.isocalendar().year.astype(int)
    daily["week"] = daily["date"].dt.isocalendar().week.astype(int)

    # Step 2: daily -> weekly
    weekly = (
        daily.groupby(["year", "week"])
        .agg(
            avg_humidity=("rh_mean", "mean"),
            max_humidity=("rh_max",  "max"),
            min_humidity=("rh_min",  "min"),
        )
        .reset_index()
    )
    weekly = weekly[weekly["week"] <= 52].copy()
    weekly["avg_humidity"] = weekly["avg_humidity"].round(1)
    weekly["max_humidity"] = weekly["max_humidity"].round(1)
    weekly["min_humidity"] = weekly["min_humidity"].round(1)
    return weekly


# ── Resume: skip already-fetched provinces ────────────────────────────────────
already_done    = set()
existing_frames = []

if os.path.exists(OUTPUT_FILE):
    existing     = pd.read_csv(OUTPUT_FILE)
    already_done = set(existing["province"].unique())
    existing_frames.append(existing)
    print(f"Resuming: {len(already_done)} provinces already in {OUTPUT_FILE}")
    print(f"  Done: {sorted(already_done)}\n")
else:
    print(f"Starting fresh -> {OUTPUT_FILE}\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
new_frames = []
failed     = []

todo = {p: c for p, c in PROVINCE_COORDS.items() if p not in already_done}
print(f"Provinces to fetch: {len(todo)} / {len(PROVINCE_COORDS)}")
print("Note: hourly data takes slightly longer to download per province.\n")

for i, (province, (lat, lon)) in enumerate(todo.items(), 1):
    print(f"[{i:02d}/{len(todo)}] {province} ({lat}, {lon})")
    success = False

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            hourly = fetch_hourly_humidity(lat, lon)
            weekly = hourly_to_weekly(hourly)
            weekly["province"] = province
            new_frames.append(weekly[["province", "year", "week",
                                      "avg_humidity", "max_humidity", "min_humidity"]])
            print(f"        OK  {len(weekly)} weekly rows  ({len(hourly):,} hourly points processed)")
            success = True
            break
        except Exception as e:
            print(f"        attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"        waiting {RETRY_WAIT}s before retry...")
                time.sleep(RETRY_WAIT)

    if not success:
        print(f"        SKIPPED after {MAX_RETRIES} attempts")
        failed.append(province)

    time.sleep(SLEEP_SEC)

# ── Save ──────────────────────────────────────────────────────────────────────
all_frames = existing_frames + new_frames

if not all_frames:
    print("\nNo data to save.")
else:
    result = pd.concat(all_frames, ignore_index=True).sort_values(["province", "year", "week"])
    result.to_csv(OUTPUT_FILE, index=False)
    n_provinces = result["province"].nunique()
    print(f"\n{'='*55}")
    print(f"Saved {len(result):,} rows  |  {n_provinces}/32 provinces  ->  {OUTPUT_FILE}")

    if failed:
        print(f"\nWARNING: {len(failed)} province(s) failed — re-run to retry:")
        for p in failed:
            print(f"   - {p}")
    else:
        print("All provinces fetched successfully.")

    print(f"\nHumidity stats (%):")
    print(result[["avg_humidity", "max_humidity", "min_humidity"]].describe().round(1))
