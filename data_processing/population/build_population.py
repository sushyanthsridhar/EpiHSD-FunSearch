"""
build_population.py
-------------------
Processes populations_size_province.csv (DR census-based annual estimates,
2000-2021) into a clean long-format file ready to merge into input.csv.

Steps:
  1. Load raw data and map province names to match input.csv format exactly
  2. Extrapolate 2022 and 2023 using linear trend from last 5 years (2017-2021)
  3. Output province_population.csv  (province, year, population)

Output: population/province_population.csv
  columns: province, year, population

Merge into input.csv via build_input.py (already wired in).

Run:
    python build_population.py
"""

import pandas as pd
import numpy as np
import os

BASE   = os.path.dirname(os.path.abspath(__file__))
INPUT  = os.path.join(BASE, "populations_size_province.csv")
OUTPUT = os.path.join(BASE, "province_population.csv")

# ── Name mapping: raw CSV names -> input.csv province names ──────────────────
# input.csv uses format "NN Province Name" with accents and special chars
NAME_MAP = {
    "Azua"                  : "02 Azua",
    "Baoruco"               : "03 Baoruco",
    "Barahona"              : "04 Barahona",
    "Dajabon"               : "05 Dajabón",
    "Distrito Nacional"     : "01 Distrito Nacional",
    "Duarte"                : "06 Duarte",
    "El Seibo"              : "08 El Seibo",
    "Elias Pina"            : "07 Elías Piña",
    "Espaillat"             : "09 Espaillat",
    "Hato Mayor"            : "30 Hato Mayor",
    "Hermanas Mirabal"      : "19 Hermanas Mirabal",
    "Independencia"         : "10 Independencia",
    "La Altagracia"         : "11 La Altagracia",
    "La Romana"             : "12 La Romana",
    "La Vega"               : "13 La Vega",
    "Maria Trinidad Sanchez": "14 María Trinidad Sánchez",
    "Monsenor Nouel"        : "28 Monseñor Nouel",
    "Monte Cristi"          : "15 Monte Cristi",
    "Monte Plata"           : "29 Monte Plata",
    "Pedernales"            : "16 Pedernales",
    "Peravia"               : "17 Peravia",
    "Puerto Plata"          : "18 Puerto Plata",
    "Samana"                : "20 Samaná",
    "San Cristobal"         : "21 San Cristóbal",
    "San Jose de Ocoa"      : "31 San José de Ocoa",
    "San Juan"              : "22 San Juan",
    "San Pedro de Macoris"  : "23 San Pedro de Macorís",
    "Sanchez Ramirez"       : "24 Sánchez Ramírez",
    "Santiago"              : "25 Santiago",
    "Santiago Rodriguez"    : "26 Santiago Rodríguez",
    "Santo Domingo"         : "32 Santo Domingo",
    "Valverde"              : "27 Valverde",
}

TARGET_YEARS = list(range(2015, 2024))   # 2015-2023
EXTRAP_YEARS = [2022, 2023]              # years to extrapolate
TREND_WINDOW = 5                         # use last N years to fit linear trend

# ── Load raw data ─────────────────────────────────────────────────────────────
print(f"Loading {INPUT} ...")
raw = pd.read_csv(INPUT)
raw.columns = raw.columns.str.strip()
raw["Province"] = raw["Province"].str.strip()
print(f"  {len(raw)} provinces, years {raw.columns[1]} - {raw.columns[-1]}")

# ── Map province names ────────────────────────────────────────────────────────
unmapped = [p for p in raw["Province"] if p not in NAME_MAP]
if unmapped:
    print(f"\nWARNING: {len(unmapped)} province(s) not in NAME_MAP:")
    for p in unmapped:
        print(f"  '{p}'")
else:
    print("  All province names mapped successfully.")

raw["province"] = raw["Province"].map(NAME_MAP)
raw = raw.dropna(subset=["province"])

# ── Melt to long format ───────────────────────────────────────────────────────
year_cols = [c for c in raw.columns if c.isdigit()]
long = raw[["province"] + year_cols].melt(
    id_vars="province", var_name="year", value_name="population"
)
long["year"] = long["year"].astype(int)
long["population"] = long["population"].astype(int)

# Keep only years available (2000-2021)
long = long[long["year"] >= 2000].copy()

# ── Extrapolate 2022 and 2023 ─────────────────────────────────────────────────
print(f"\nExtrapolating {EXTRAP_YEARS} using {TREND_WINDOW}-year linear trend ...")

extrap_rows = []
for prov in long["province"].unique():
    sub = long[long["province"] == prov].sort_values("year")
    # Fit linear trend on last TREND_WINDOW years
    trend_data = sub.tail(TREND_WINDOW)
    x = trend_data["year"].values
    y = trend_data["population"].values
    slope, intercept = np.polyfit(x, y, 1)

    for yr in EXTRAP_YEARS:
        pop_est = int(round(slope * yr + intercept))
        extrap_rows.append({"province": prov, "year": yr, "population": pop_est})
        print(f"  {prov:35s}  {yr}: {pop_est:,}  (trend slope: {slope:+.0f}/yr)")

extrap_df = pd.DataFrame(extrap_rows)
full = pd.concat([long, extrap_df], ignore_index=True)

# ── Filter to target years only ───────────────────────────────────────────────
full = full[full["year"].isin(TARGET_YEARS)].sort_values(["province","year"]).reset_index(drop=True)

# ── Validate ──────────────────────────────────────────────────────────────────
print(f"\n=== Validation ===")
print(f"Provinces  : {full['province'].nunique()} / 32")
print(f"Years      : {sorted(full['year'].unique())}")
print(f"Total rows : {len(full)} (expect {32 * len(TARGET_YEARS)} = {32*len(TARGET_YEARS)})")
print(f"NaNs       : {full.isnull().sum().sum()}")

missing_provs = set(NAME_MAP.values()) - set(full["province"].unique())
if missing_provs:
    print(f"Missing provinces: {missing_provs}")
else:
    print("All 32 provinces present.")

# ── Save ──────────────────────────────────────────────────────────────────────
full.to_csv(OUTPUT, index=False)
print(f"\nSaved -> {OUTPUT}")

# ── Preview ───────────────────────────────────────────────────────────────────
print("\n=== 2023 population by province (largest to smallest) ===")
pop2023 = full[full["year"]==2023].sort_values("population", ascending=False)
print(pop2023[["province","population"]].to_string(index=False))
