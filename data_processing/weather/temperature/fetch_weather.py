"""
fetch_weather.py
----------------
Fetches weekly temperature data (avg, max, min) for each Dominican Republic province
from the Open-Meteo Historical Weather API (ERA5 reanalysis, free, no API key).

Output: weekly_weather_by_province.csv
    columns: province, year, week, avg_temp, max_temp, min_temp

Merge usage:
    import pandas as pd
    cases   = pd.read_csv("timeseries_all_provinces.csv")
    weather = pd.read_csv("weekly_weather_by_province.csv")
    df      = cases.merge(weather, on=["province", "year", "week"], how="left")

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
OUTPUT_FILE = "weekly_weather_by_province.csv"
SLEEP_SEC   = 1.0    # pause between provinces (be polite to the API)
MAX_RETRIES = 3      # retry each province this many times on failure
RETRY_WAIT  = 5.0    # seconds to wait between retries


def fetch_daily_temps(lat: float, lon: float) -> pd.DataFrame:
    """Call Open-Meteo and return a DataFrame with daily temperature columns."""
    params = {
        "latitude"  : lat,
        "longitude" : lon,
        "start_date": START_DATE,
        "end_date"  : END_DATE,
        "daily"     : "temperature_2m_max,temperature_2m_min,temperature_2m_mean",
        "timezone"  : "America/Santo_Domingo",
    }
    resp = requests.get(API_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()["daily"]

    df = pd.DataFrame({
        "date"     : pd.to_datetime(data["time"]),
        "tmax_day" : data["temperature_2m_max"],
        "tmin_day" : data["temperature_2m_min"],
        "tmean_day": data["temperature_2m_mean"],
    })
    return df


def daily_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily temps to ISO week (matching the cases dataset convention)."""
    daily = daily.copy()
    daily["year"] = daily["date"].dt.isocalendar().year.astype(int)
    daily["week"] = daily["date"].dt.isocalendar().week.astype(int)

    weekly = (
        daily.groupby(["year", "week"])
        .agg(
            avg_temp=("tmean_day", "mean"),
            max_temp=("tmax_day",  "max"),
            min_temp=("tmin_day",  "min"),
        )
        .reset_index()
    )
    # Keep only weeks 1-52 to match the zero-filled cases grid
    weekly = weekly[weekly["week"] <= 52].copy()
    weekly["avg_temp"] = weekly["avg_temp"].round(2)
    weekly["max_temp"] = weekly["max_temp"].round(2)
    weekly["min_temp"] = weekly["min_temp"].round(2)
    return weekly


# ── Load any previously fetched provinces (re-run safety) ────────────────────
already_done = set()
existing_frames = []

if os.path.exists(OUTPUT_FILE):
    existing = pd.read_csv(OUTPUT_FILE)
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
print(f"Provinces to fetch: {len(todo)} / {len(PROVINCE_COORDS)}\n")

for i, (province, (lat, lon)) in enumerate(todo.items(), 1):
    print(f"[{i:02d}/{len(todo)}] {province} ({lat}, {lon})")
    success = False

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            daily  = fetch_daily_temps(lat, lon)
            weekly = daily_to_weekly(daily)
            weekly["province"] = province
            new_frames.append(weekly[["province", "year", "week", "avg_temp", "max_temp", "min_temp"]])
            print(f"        OK  {len(weekly)} weekly rows")
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


# ── Save (merge with any previously saved data) ───────────────────────────────
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
        print(f"\nWARNING: {len(failed)} province(s) failed - re-run the script to retry:")
        for p in failed:
            print(f"   - {p}")
    else:
        print("All provinces fetched successfully.")

    print(f"\nTemp stats:")
    print(result[["avg_temp", "max_temp", "min_temp"]].describe().round(2))
