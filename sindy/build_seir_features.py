"""
build_seir_features.py
────────────────────────────────────────────────────────────────────────────
Reads data_processing/input.csv and engineers SEIR-relevant features for
SINDy equation discovery.

Output: sindy/seir_features.csv

Features added (all computed per-province, respecting temporal order):
────────────────────────────────────────────────────────────────────────────
  S_proxy          – (population − cumulative_cases) / population  ∈ [0,1]
                     Proxy for susceptible fraction S(t)/N
  dI_dt            – Δtotal_cases week-over-week  (forward: I[t] − I[t-1])
                     RHS target variable for the I equation in SEIR
  cases_lag1       – total_cases at t-1 (1 week lag)
  cases_lag2       – total_cases at t-2 (2 week lag)
  cases_lag4       – total_cases at t-4 (4 week lag, ~1 serial interval)
  rolling_cases_4wk– 4-week centred rolling mean of total_cases
  rolling_cases_8wk– 8-week rolling mean (slow epidemic trend)
  EIP              – exp(0.347·avg_temp − 9.17)
                     Extrinsic Incubation Period thermal function
                     (higher T → shorter EIP → faster transmission)
  temp_lag4        – avg_temp at t-4 (aligns with ~4-week EIP at 28°C)
  rainfall_lag3    – avg_rainfall at t-3 (3-week lag: rain → breeding → bite)
  week_sin         – sin(2π·week/52)  — cyclical seasonal forcing
  week_cos         – cos(2π·week/52)
────────────────────────────────────────────────────────────────────────────
Rows with NaN (early weeks that cannot compute lags) are dropped per province.
A validation summary is printed at the end.
"""

import os
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV  = os.path.join(BASE, "..", "data_processing", "input.csv")
OUTPUT_CSV = os.path.join(BASE, "seir_features.csv")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading input.csv …")
df = pd.read_csv(INPUT_CSV)
print(f"  Shape: {df.shape}  |  Provinces: {df['province'].nunique()}")
df = df.sort_values(["province", "year", "week"]).reset_index(drop=True)

# ── Per-province feature engineering ─────────────────────────────────────────
frames = []
provinces = df["province"].unique()

for prov in provinces:
    sub = df[df["province"] == prov].copy().reset_index(drop=True)

    # ── Cumulative cases & susceptible proxy ──────────────────────────────────
    sub["cumulative_cases"] = sub["total_cases"].cumsum()
    # Clamp so S_proxy never goes negative (can happen if reported cases exceed
    # census population due to undercount in denominator)
    sub["S_proxy"] = ((sub["population"] - sub["cumulative_cases"]) /
                       sub["population"]).clip(lower=0.0)

    # ── Rate of change of infected compartment ────────────────────────────────
    sub["dI_dt"] = sub["total_cases"].diff(1)          # I[t] − I[t-1]

    # ── Case lags ─────────────────────────────────────────────────────────────
    sub["cases_lag1"] = sub["total_cases"].shift(1)
    sub["cases_lag2"] = sub["total_cases"].shift(2)
    sub["cases_lag4"] = sub["total_cases"].shift(4)

    # ── Rolling averages (min_periods avoids NaN for short tails) ─────────────
    sub["rolling_cases_4wk"] = (sub["total_cases"]
                                  .rolling(4, min_periods=2).mean())
    sub["rolling_cases_8wk"] = (sub["total_cases"]
                                  .rolling(8, min_periods=4).mean())

    # ── EIP thermal function ──────────────────────────────────────────────────
    # EIP(T) = exp(0.347·T − 9.17)   [Brady et al. 2013]
    # Bounded to [0.01, 20] to avoid numerical blow-up at extreme temps
    sub["EIP"] = np.exp(0.347 * sub["avg_temp"] - 9.17).clip(0.01, 20.0)

    # ── Environmental lags ────────────────────────────────────────────────────
    sub["temp_lag4"]     = sub["avg_temp"].shift(4)
    sub["rainfall_lag3"] = sub["avg_rainfall"].shift(3)

    # ── Cyclical seasonal encoding ────────────────────────────────────────────
    sub["week_sin"] = np.sin(2 * np.pi * sub["week"] / 52)
    sub["week_cos"] = np.cos(2 * np.pi * sub["week"] / 52)

    # ── Normalised incidence (per 100,000 population) ─────────────────────────
    # All case-based features are re-expressed as rates so that the SINDy
    # equation is population-invariant and transferable across provinces of
    # different sizes.  Coefficients then carry units of weeks⁻¹ × 100k⁻¹.
    pop = sub["population"]
    sub["incidence"]             = sub["total_cases"]        / pop * 1e5
    sub["dI_dt_norm"]            = sub["dI_dt"]              / pop * 1e5
    sub["incidence_lag1"]        = sub["cases_lag1"]         / pop * 1e5
    sub["incidence_lag2"]        = sub["cases_lag2"]         / pop * 1e5
    sub["incidence_lag4"]        = sub["cases_lag4"]         / pop * 1e5
    sub["rolling_incidence_4wk"] = sub["rolling_cases_4wk"]  / pop * 1e5

    frames.append(sub)

# ── Concatenate & drop NaN rows ───────────────────────────────────────────────
out = pd.concat(frames, ignore_index=True)

# Count NaNs before dropping (for reporting)
nan_before = out.isnull().sum().sum()
rows_before = len(out)

out = out.dropna().reset_index(drop=True)
rows_dropped = rows_before - len(out)

# ── Column ordering ───────────────────────────────────────────────────────────
col_order = [
    "province", "year", "month", "week",
    "total_cases", "population",
    # SEIR proxies (raw)
    "cumulative_cases", "S_proxy", "dI_dt",
    # Lags (raw)
    "cases_lag1", "cases_lag2", "cases_lag4",
    # Rolling (raw)
    "rolling_cases_4wk", "rolling_cases_8wk",
    # Normalised incidence (per 100k) — use these as SINDy features
    "incidence", "dI_dt_norm",
    "incidence_lag1", "incidence_lag2", "incidence_lag4",
    "rolling_incidence_4wk",
    # Thermal
    "avg_temp", "max_temp", "min_temp", "EIP", "temp_lag4",
    # Rainfall
    "avg_rainfall", "max_rainfall", "min_rainfall", "rainfall_lag3",
    # Humidity
    "avg_humidity", "max_humidity", "min_humidity",
    # Seasonality
    "week_sin", "week_cos",
]
out = out[col_order]

# ── Save ──────────────────────────────────────────────────────────────────────
out.to_csv(OUTPUT_CSV, index=False)

# ── Validation summary ────────────────────────────────────────────────────────
print("\n=== seir_features.csv — Validation Summary ===")
print(f"Output rows   : {len(out):,}  (dropped {rows_dropped} NaN rows, {nan_before} total NaN cells before drop)")
print(f"Columns       : {len(out.columns)}  → {list(out.columns)}")
print(f"Provinces     : {out['province'].nunique()}  (expected 32)")
print(f"NaNs remaining: {out.isnull().sum().sum()}  (should be 0)")
print(f"Years covered : {sorted(out['year'].unique())}")

print("\n--- S_proxy ---")
print(f"  Range: [{out['S_proxy'].min():.4f}, {out['S_proxy'].max():.4f}]")
print(f"  Mean : {out['S_proxy'].mean():.4f}  (expect close to 1.0 — dengue rarely exhausts susceptibles)")
print(f"  Rows with S_proxy < 0.95: {(out['S_proxy'] < 0.95).sum()}")

print("\n--- dI_dt (weekly case change, raw) ---")
print(f"  Range: [{out['dI_dt'].min():.1f}, {out['dI_dt'].max():.1f}]")
print(f"  Mean : {out['dI_dt'].mean():.2f}  (expect ~0, rises balanced by falls)")

print("\n--- dI_dt_norm (normalised per 100k) ---")
print(f"  Range: [{out['dI_dt_norm'].min():.3f}, {out['dI_dt_norm'].max():.3f}]")
print(f"  Mean : {out['dI_dt_norm'].mean():.4f}")

print("\n--- incidence (per 100k) ---")
print(f"  Range: [{out['incidence'].min():.3f}, {out['incidence'].max():.3f}]")
print(f"  Mean : {out['incidence'].mean():.4f}")

print("\n--- EIP (thermal forcing) ---")
print(f"  Range: [{out['EIP'].min():.4f}, {out['EIP'].max():.4f}]")
typical_temps = [20, 25, 28, 32, 35]
for t in typical_temps:
    print(f"    EIP({t}°C) = {np.exp(0.347*t - 9.17):.4f}")

print("\n--- Seasonal encoding sanity ---")
print(f"  week_sin range: [{out['week_sin'].min():.3f}, {out['week_sin'].max():.3f}]  (expect [-1, 1])")
print(f"  week_cos range: [{out['week_cos'].min():.3f}, {out['week_cos'].max():.3f}]  (expect [-1, 1])")

print("\n--- Rows per province (all should be equal) ---")
prov_counts = out.groupby("province").size()
print(f"  Min: {prov_counts.min()}  Max: {prov_counts.max()}  Mean: {prov_counts.mean():.1f}")
if prov_counts.min() != prov_counts.max():
    print("  WARN: unequal rows across provinces — check lags at year boundaries")
else:
    print("  PASS: all provinces have equal row count")

print(f"\nSaved → {OUTPUT_CSV}")
