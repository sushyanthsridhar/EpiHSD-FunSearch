"""
validate_humidity.py
--------------------
Validates weekly_humidity_by_province.csv across 8 checks.

Run with:
    python validate_humidity.py
"""

import pandas as pd
import numpy as np
import sys

HUMIDITY_FILE          = "weekly_humidity_by_province.csv"
EXPECTED_PROVINCES     = 32
EXPECTED_ROWS_PER_PROV = 468   # 9 years x 52 weeks

# DR geography: most humid provinces
# - Samaná, Puerto Plata (wet Atlantic coast)
# - Monseñor Nouel, María Trinidad Sánchez (orographic moisture)
HUMID_PROVINCES = {"20 Samaná", "18 Puerto Plata",
                   "28 Monseñor Nouel", "14 María Trinidad Sánchez"}

# DR geography: driest/least humid provinces
# - Pedernales, Independencia (arid southwest rain shadow)
# - Monte Cristi (arid northwest)
DRY_PROVINCES = {"16 Pedernales", "10 Independencia", "15 Monte Cristi"}

# Rainy season = higher humidity
# DR rainy seasons: weeks 17-22 (May-June) and 35-45 (Aug-Nov)
RAINY_WEEKS = set(range(17, 23)) | set(range(35, 46))
DRY_WEEKS   = set(range(1, 14)) | set(range(49, 53))

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

print(f"\nLoading {HUMIDITY_FILE} ...")
try:
    df = pd.read_csv(HUMIDITY_FILE)
    print(f"  Shape: {df.shape}\n")
except FileNotFoundError:
    print(f"  ERROR: file not found. Run fetch_humidity.py first.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("CHECK 1 — Province count")
n_prov = df["province"].nunique()
if n_prov == EXPECTED_PROVINCES:
    check("Province count", PASS, f"{n_prov} / {EXPECTED_PROVINCES}")
else:
    check("Province count", FAIL,
          f"Got {n_prov} / {EXPECTED_PROVINCES} — re-run fetch_humidity.py")

# ─────────────────────────────────────────────────────────────────────────────
print("\nCHECK 2 — Row count per province (expect 468)")
counts = df.groupby("province").size()
bad    = counts[counts != EXPECTED_ROWS_PER_PROV]
if len(bad) == 0:
    check("Row counts", PASS, f"All {n_prov} provinces have {EXPECTED_ROWS_PER_PROV} rows")
else:
    check("Row counts", FAIL, f"{len(bad)} province(s) have wrong row count:")
    print(bad.to_string())

# ─────────────────────────────────────────────────────────────────────────────
print("\nCHECK 3 — No missing values")
nans = df.isnull().sum()
if nans.sum() == 0:
    check("No NaNs", PASS, "Zero missing values")
else:
    check("No NaNs", FAIL, f"{nans.sum()} missing values:")
    print(nans[nans > 0].to_string())

# ─────────────────────────────────────────────────────────────────────────────
print("\nCHECK 4 — No duplicate (province, year, week)")
dups = df.duplicated(subset=["province", "year", "week"]).sum()
if dups == 0:
    check("No duplicates", PASS)
else:
    check("No duplicates", FAIL, f"{dups} duplicate rows found")

# ─────────────────────────────────────────────────────────────────────────────
print("\nCHECK 5 — Ordering: min <= avg <= max")
bad_order = df[(df["min_humidity"] > df["avg_humidity"]) |
               (df["avg_humidity"] > df["max_humidity"])]
if len(bad_order) == 0:
    check("Ordering (min<=avg<=max)", PASS, "Holds for all rows")
else:
    check("Ordering (min<=avg<=max)", FAIL,
          f"{len(bad_order)} rows violate ordering")
    print(bad_order.head(5).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
print("\nCHECK 6 — Physical plausibility (RH must be 0-100%)")
out_low  = (df[["avg_humidity","max_humidity","min_humidity"]] < 0).sum().sum()
out_high = (df[["avg_humidity","max_humidity","min_humidity"]] > 100).sum().sum()
# For DR tropics: avg should realistically be 50-95%
implaus  = df[(df["avg_humidity"] < 40) | (df["avg_humidity"] > 99)]

if out_low == 0 and out_high == 0:
    check("Hard bounds (0-100%)", PASS, "All values within valid RH range")
else:
    check("Hard bounds (0-100%)", FAIL,
          f"{out_low} values <0%  |  {out_high} values >100%")

if len(implaus) == 0:
    check("Tropical plausibility (40-99%)", PASS,
          "All avg_humidity values within expected tropical range")
else:
    check("Tropical plausibility (40-99%)", WARN,
          f"{len(implaus)} rows outside 40-99% — inspect:")
    print(implaus.groupby("province")[["avg_humidity"]].agg(["min","max"]).to_string())

# ─────────────────────────────────────────────────────────────────────────────
print("\nCHECK 7 — Seasonal pattern (rainy season more humid than dry season)")
rainy_avg = df[df["week"].isin(RAINY_WEEKS)].groupby("province")["avg_humidity"].mean()
dry_avg   = df[df["week"].isin(DRY_WEEKS)].groupby("province")["avg_humidity"].mean()

both  = pd.DataFrame({"rainy": rainy_avg, "dry": dry_avg}).dropna()
fails = both[both["rainy"] <= both["dry"]]

if len(fails) == 0:
    diff = (both["rainy"] - both["dry"]).mean()
    check("Seasonal pattern", PASS,
          f"Rainy seasons {diff:.1f}% more humid than dry season on average")
else:
    check("Seasonal pattern", WARN,
          f"{len(fails)} province(s) not more humid in rainy season:")
    print(fails.round(1).to_string())

# ─────────────────────────────────────────────────────────────────────────────
print("\nCHECK 8 — Geographic contrast (humid coast > arid southwest)")
prov_means   = df.groupby("province")["avg_humidity"].mean()
humid_present = HUMID_PROVINCES & set(prov_means.index)
dry_present   = DRY_PROVINCES   & set(prov_means.index)

if humid_present and dry_present:
    humid_avg = prov_means[list(humid_present)].mean()
    dry_avg_g = prov_means[list(dry_present)].mean()
    diff = humid_avg - dry_avg_g
    if diff > 0:
        check("Humid/dry geographic contrast", PASS,
              f"Humid coast avg {humid_avg:.1f}%  vs  Arid SW avg {dry_avg_g:.1f}%  (diff={diff:.1f}%)")
    else:
        check("Humid/dry geographic contrast", WARN,
              f"Expected coast to be more humid — check centroids or data")
else:
    check("Humid/dry geographic contrast", WARN, "Missing reference provinces")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
n_pass = results.count(PASS)
n_fail = results.count(FAIL)
n_warn = results.count(WARN)
print(f"Summary: {n_pass} PASS  |  {n_warn} WARN  |  {n_fail} FAIL  (out of {len(results)} checks)")

if n_fail > 0:
    print("\nAction: fix FAIL items — likely need to re-run fetch_humidity.py")
elif n_warn > 0:
    print("\nLooks good. Review WARN items before merging into input.csv.")
else:
    print("\nAll checks passed. Safe to merge into input.csv.")

print()

# ─────────────────────────────────────────────────────────────────────────────
print("=== REFERENCE: Per-province avg_humidity ranking ===")
print(df.groupby("province")["avg_humidity"].mean().sort_values(ascending=False).round(1).to_string())
print()
print("=== REFERENCE: Weekly seasonal pattern (all provinces avg) ===")
print(df.groupby("week")["avg_humidity"].mean().round(1).to_string())
