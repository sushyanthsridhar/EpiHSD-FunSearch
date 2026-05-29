"""
validate_rainfall.py
--------------------
Validates weekly_rainfall_by_province.csv across 8 checks.

Run with:
    python validate_rainfall.py
"""

import pandas as pd
import numpy as np
import sys

RAINFALL_FILE           = "weekly_rainfall_by_province.csv"
EXPECTED_PROVINCES      = 32
EXPECTED_ROWS_PER_PROV  = 468   # 9 years x 52 weeks

# DR geography: wettest provinces (northeast/central, Atlantic-facing slopes)
WET_PROVINCES = {"28 Monseñor Nouel", "14 María Trinidad Sánchez",
                 "09 Espaillat", "13 La Vega", "20 Samaná"}
# DR geography: driest provinces (southwest rain shadow + arid northwest)
DRY_PROVINCES = {"16 Pedernales", "10 Independencia",
                 "15 Monte Cristi", "07 Elías Piña"}

# DR has TWO rainy seasons:
#   1st: ~weeks 17-22 (May-early June)
#   2nd: ~weeks 35-45 (late Aug - Nov, hurricane season)
# Dry seasons: weeks 1-13 (Jan-Mar) and weeks 48-52 (Dec)
RAINY_WEEKS = set(range(17, 23)) | set(range(35, 46))
DRY_WEEKS   = set(range(1, 14)) | set(range(49, 53))

# Physically, a single daily value above 300mm is extremely rare even in hurricanes
MAX_PLAUSIBLE_DAILY_MM = 300

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

print(f"\nLoading {RAINFALL_FILE} ...")
try:
    df = pd.read_csv(RAINFALL_FILE)
    print(f"  Shape: {df.shape}\n")
except FileNotFoundError:
    print(f"  ERROR: file not found. Run fetch_rainfall.py first.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("CHECK 1 — Province count")
n_prov = df["province"].nunique()
if n_prov == EXPECTED_PROVINCES:
    check("Province count", PASS, f"{n_prov} / {EXPECTED_PROVINCES}")
else:
    check("Province count", FAIL, f"Got {n_prov} / {EXPECTED_PROVINCES} — re-run fetch_rainfall.py")

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
print("\nCHECK 5 — No negative rainfall, min <= avg <= max")
neg = (df[["avg_rainfall","max_rainfall","min_rainfall"]] < 0).sum().sum()
bad_order = df[(df["min_rainfall"] > df["avg_rainfall"]) |
               (df["avg_rainfall"] > df["max_rainfall"])]
if neg == 0 and len(bad_order) == 0:
    check("No negatives & ordering", PASS, "min <= avg <= max, all values >= 0")
else:
    if neg > 0:
        check("No negatives", FAIL, f"{neg} negative rainfall values found")
    if len(bad_order) > 0:
        check("Ordering (min<=avg<=max)", FAIL, f"{len(bad_order)} rows violate ordering")
        print(bad_order.head(5).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
print("\nCHECK 6 — Physical plausibility (max daily <= 300mm)")
extreme = df[df["max_rainfall"] > MAX_PLAUSIBLE_DAILY_MM]
if len(extreme) == 0:
    check("Physical plausibility", PASS,
          f"All max_rainfall values below {MAX_PLAUSIBLE_DAILY_MM} mm/day")
else:
    check("Physical plausibility", WARN,
          f"{len(extreme)} week(s) exceed {MAX_PLAUSIBLE_DAILY_MM} mm/day — likely hurricane events:")
    print(extreme[["province","year","week","avg_rainfall","max_rainfall"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
print("\nCHECK 7 — Bimodal seasonal pattern (DR has 2 rainy seasons)")
# Rainy season weeks should be wetter than dry season weeks
rainy_avg = df[df["week"].isin(RAINY_WEEKS)].groupby("province")["avg_rainfall"].mean()
dry_avg   = df[df["week"].isin(DRY_WEEKS)].groupby("province")["avg_rainfall"].mean()

both = pd.DataFrame({"rainy": rainy_avg, "dry": dry_avg}).dropna()
fails = both[both["rainy"] <= both["dry"]]

if len(fails) == 0:
    diff = (both["rainy"] - both["dry"]).mean()
    check("Bimodal seasonality", PASS,
          f"Rainy seasons {diff:.2f} mm/day wetter than dry season across all provinces")
else:
    check("Bimodal seasonality", WARN,
          f"{len(fails)} province(s) don't show wetter rainy seasons:")
    print(fails.round(2).to_string())

# ─────────────────────────────────────────────────────────────────────────────
print("\nCHECK 8 — Geographic wet/dry contrast (northeast wetter than southwest)")
prov_means = df.groupby("province")["avg_rainfall"].mean()

wet_present = WET_PROVINCES & set(prov_means.index)
dry_present = DRY_PROVINCES & set(prov_means.index)

if wet_present and dry_present:
    wet_avg = prov_means[list(wet_present)].mean()
    dry_avg_geo = prov_means[list(dry_present)].mean()
    diff = wet_avg - dry_avg_geo
    if diff > 0:
        check("Wet/dry geographic contrast", PASS,
              f"Wet provinces avg {wet_avg:.2f} mm/day  vs  Dry provinces avg {dry_avg_geo:.2f} mm/day  (diff={diff:.2f})")
    else:
        check("Wet/dry geographic contrast", WARN,
              f"Expected wet provinces to be wetter — check centroids")
else:
    check("Wet/dry geographic contrast", WARN, "Missing some reference provinces")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
n_pass = results.count(PASS)
n_fail = results.count(FAIL)
n_warn = results.count(WARN)
print(f"Summary: {n_pass} PASS  |  {n_warn} WARN  |  {n_fail} FAIL  (out of {len(results)} checks)")

if n_fail > 0:
    print("\nAction: fix FAIL items — likely need to re-run fetch_rainfall.py")
elif n_warn > 0:
    print("\nLooks good. Review WARN items (likely hurricane extreme events — expected).")
else:
    print("\nAll checks passed. Safe to merge into input.csv.")
print()
