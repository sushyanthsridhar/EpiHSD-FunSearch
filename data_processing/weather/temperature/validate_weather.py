"""
validate_weather.py
-------------------
Runs a series of checks on weekly_weather_by_province.csv to confirm
the fetch was complete and the data is physically plausible.

Checks:
  1. Province count          — must be exactly 32
  2. Row count per province  — must be 468 (9 years x 52 weeks)
  3. No missing values       — no NaNs anywhere
  4. No duplicate rows       — (province, year, week) must be unique
  5. Temperature ordering    — min_temp <= avg_temp <= max_temp always
  6. Physical plausibility   — DR is tropical: temps should be 15-40 C
  7. Seasonal pattern        — avg_temp should peak in summer (weeks 26-39)
  8. Cross-province sanity   — highland provinces cooler than coastal ones

Run with:
    python validate_weather.py
"""

import pandas as pd
import numpy as np
import sys

WEATHER_FILE = "weekly_weather_by_province.csv"
EXPECTED_PROVINCES = 32
EXPECTED_ROWS_PER_PROV = 468   # 9 years * 52 weeks

# Known geography: highland provinces should be cooler than coastal/lowland ones
# Cordillera Central = La Vega, San Jose de Ocoa, Monsenor Nouel (higher elevation)
# Coastal/hot        = Barahona, Pedernales, San Pedro de Macoris
HIGHLAND_PROVINCES = {"13 La Vega", "31 San José de Ocoa", "28 Monseñor Nouel"}
COASTAL_PROVINCES  = {"04 Barahona", "16 Pedernales", "23 San Pedro de Macorís"}

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

results = []

def check(name, status, detail=""):
    tag = f"[{status}]"
    print(f"  {tag:8s} {name}")
    if detail:
        print(f"           {detail}")
    results.append(status)

print(f"\nLoading {WEATHER_FILE} ...")
try:
    df = pd.read_csv(WEATHER_FILE)
    print(f"  Shape: {df.shape}\n")
except FileNotFoundError:
    print(f"  ERROR: file not found. Run fetch_weather.py first.")
    sys.exit(1)

print("=" * 55)
print("CHECK 1 — Province count")
n_prov = df["province"].nunique()
if n_prov == EXPECTED_PROVINCES:
    check("Province count", PASS, f"{n_prov} / {EXPECTED_PROVINCES}")
else:
    missing = set(
        [f"{i:02d}" for i in range(1, 33)]
    )
    found_ids = {p.split()[0] for p in df["province"].unique()}
    missing_provs = [p for p in df["province"].unique() if p not in df["province"].unique()]
    check("Province count", FAIL,
          f"Got {n_prov} / {EXPECTED_PROVINCES} — re-run fetch_weather.py to get the rest")
    print(f"           Provinces present: {sorted(df['province'].unique())}")

print("\nCHECK 2 — Row count per province (expect {})".format(EXPECTED_ROWS_PER_PROV))
row_counts = df.groupby("province").size()
bad = row_counts[row_counts != EXPECTED_ROWS_PER_PROV]
if len(bad) == 0:
    check("Row counts", PASS, f"All {n_prov} provinces have {EXPECTED_ROWS_PER_PROV} rows")
else:
    check("Row counts", FAIL, f"{len(bad)} province(s) have wrong row count:")
    print(bad.to_string())

print("\nCHECK 3 — No missing values")
nans = df.isnull().sum()
total_nans = nans.sum()
if total_nans == 0:
    check("No NaNs", PASS, "Zero missing values")
else:
    check("No NaNs", FAIL, f"{total_nans} missing values found:")
    print(nans[nans > 0].to_string())

print("\nCHECK 4 — No duplicate (province, year, week)")
dups = df.duplicated(subset=["province", "year", "week"]).sum()
if dups == 0:
    check("No duplicates", PASS)
else:
    check("No duplicates", FAIL, f"{dups} duplicate rows found")

print("\nCHECK 5 — Temperature ordering (min <= avg <= max)")
bad_order = df[(df["min_temp"] > df["avg_temp"]) | (df["avg_temp"] > df["max_temp"])]
if len(bad_order) == 0:
    check("Temp ordering", PASS, "min <= avg <= max holds for all rows")
else:
    check("Temp ordering", FAIL, f"{len(bad_order)} rows violate min <= avg <= max")
    print(bad_order.head(5).to_string(index=False))

print("\nCHECK 6 — Physical plausibility (DR tropics: 15–40 °C)")
out_of_range = df[
    (df["min_temp"] < 10) | (df["min_temp"] > 35) |
    (df["avg_temp"] < 15) | (df["avg_temp"] > 38) |
    (df["max_temp"] < 18) | (df["max_temp"] > 42)
]
if len(out_of_range) == 0:
    check("Plausible range", PASS, f"All values within expected tropical bounds")
else:
    check("Plausible range", WARN,
          f"{len(out_of_range)} rows outside expected range (15-40 C) — inspect:")
    print(out_of_range.groupby("province")[["avg_temp","max_temp","min_temp"]].agg(["min","max"]).to_string())

print("\nCHECK 7 — Seasonal pattern (summer warmer than winter)")
df["season"] = df["week"].apply(lambda w: "summer" if 26 <= w <= 39 else
                                           "winter" if w <= 8 or w >= 47 else "other")
seasonal = df.groupby(["province", "season"])["avg_temp"].mean().unstack()
if "summer" in seasonal.columns and "winter" in seasonal.columns:
    bad_seasonal = seasonal[seasonal["summer"] <= seasonal["winter"]]
    if len(bad_seasonal) == 0:
        avg_diff = (seasonal["summer"] - seasonal["winter"]).mean()
        check("Seasonal pattern", PASS,
              f"Summer avg {avg_diff:.2f} C warmer than winter across all provinces")
    else:
        check("Seasonal pattern", WARN,
              f"{len(bad_seasonal)} province(s) don't show warmer summers:")
        print(bad_seasonal[["summer","winter"]].round(2).to_string())
else:
    check("Seasonal pattern", WARN, "Could not compute — check week column values")

print("\nCHECK 8 — Cross-province sanity (highlands cooler than coast)")
prov_means = df.groupby("province")["avg_temp"].mean()
highland_present = HIGHLAND_PROVINCES & set(prov_means.index)
coastal_present  = COASTAL_PROVINCES  & set(prov_means.index)

if highland_present and coastal_present:
    highland_avg = prov_means[list(highland_present)].mean()
    coastal_avg  = prov_means[list(coastal_present)].mean()
    diff = coastal_avg - highland_avg
    if diff > 0:
        check("Highland/coastal contrast", PASS,
              f"Coastal avg {coastal_avg:.2f} C  vs  Highland avg {highland_avg:.2f} C  (diff={diff:.2f} C)")
    else:
        check("Highland/coastal contrast", WARN,
              f"Highlands ({highland_avg:.2f} C) not cooler than coast ({coastal_avg:.2f} C) — check centroids")
else:
    check("Highland/coastal contrast", WARN, "Missing some provinces to compare")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
n_pass = results.count(PASS)
n_fail = results.count(FAIL)
n_warn = results.count(WARN)
print(f"Summary: {n_pass} PASS  |  {n_warn} WARN  |  {n_fail} FAIL  (out of {len(results)} checks)")

if n_fail > 0:
    print("\nAction: fix FAIL items first — likely need to re-run fetch_weather.py")
elif n_warn > 0:
    print("\nLooks mostly good. Review WARN items before merging into final dataset.")
else:
    print("\nAll checks passed. Safe to merge into your main dataset.")

print()
