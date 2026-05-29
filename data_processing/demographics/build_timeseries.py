import pandas as pd
import os
from itertools import product

INPUT_FILE  = "cases_by_province_week.csv"
PROV_DIR    = "timeseries_by_province"
COMBINED    = "timeseries_all_provinces.csv"

os.makedirs(PROV_DIR, exist_ok=True)

# ── 1. Load & clean ───────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE, low_memory=False)
df = df[df['year'] != 2014]
df = df[df['province'] != '99 Extranjero']

# ── 2. Resolve duplicate province+year+week rows ──────────────────────────────
# Some weeks span two months — sum cases, keep the higher month number
df = (df.groupby(['province', 'year', 'week'], as_index=False)
        .agg(month=('month', 'max'), total_cases=('total_cases', 'sum')))

df = df.sort_values(['province', 'year', 'week']).reset_index(drop=True)

provinces = sorted(df['province'].unique())
all_years  = sorted(df['year'].unique())
all_weeks  = range(1, 53)

print(f"Provinces: {len(provinces)}, Years: {all_years}")
print(f"Duplicates after fix: {df.duplicated(subset=['province','year','week']).sum()}")

# ── 3. Full weekly grid per province (zero-fill gaps) ─────────────────────────
full_index = pd.DataFrame(list(product(all_years, all_weeks)), columns=['year','week'])
combined_frames = []

for prov in provinces:
    sub = df[df['province'] == prov][['year','week','month','total_cases']]

    full = full_index.merge(sub, on=['year','week'], how='left')
    full['total_cases'] = full['total_cases'].fillna(0).astype(int)
    full['month']       = full['month'].ffill().fillna(0).astype(int)
    full['province']    = prov
    full = full[['province','year','month','week','total_cases']]
    full = full.sort_values(['year','week']).reset_index(drop=True)

    safe  = prov.replace('/','_').replace(' ','_')
    fpath = os.path.join(PROV_DIR, f"{safe}.csv")
    full.to_csv(fpath, index=False)
    print(f"  Saved: {fpath}  ({len(full)} rows)")

    combined_frames.append(full)

# ── 4. Combined CSV ───────────────────────────────────────────────────────────
combined = pd.concat(combined_frames, ignore_index=True)
combined = combined.sort_values(['province','year','week']).reset_index(drop=True)
combined.to_csv(COMBINED, index=False)

# ── 5. Final validation ───────────────────────────────────────────────────────
dups  = combined.duplicated(subset=['province','year','week']).sum()
nans  = combined.isnull().sum().sum()
zeros_month = (combined['month']==0).sum()
expected = len(provinces) * len(all_years) * 52

print(f"\n=== Validation ===")
print(f"Total rows     : {len(combined)}  (expected {expected})")
print(f"Duplicates     : {dups}  (should be 0)")
print(f"NaNs           : {nans}  (should be 0)")
print(f"Month=0 rows   : {zeros_month}  (should be 0)")
print(f"Negative cases : {(combined['total_cases']<0).sum()}  (should be 0)")
print(f"\nRows per province (all should be {len(all_years)*52}):")
print(combined.groupby('province').size().to_string())