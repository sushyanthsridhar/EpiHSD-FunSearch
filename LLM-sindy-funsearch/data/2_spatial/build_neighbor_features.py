"""
build_neighbor_features.py
────────────────────────────────────────────────────────────────
Computes spatial neighbor incidence features.

For each province and each week, this script finds all neighboring
provinces (within 150km) and computes a distance-weighted average
of their dengue incidence. This captures cross-province spillover,
the idea that outbreaks in nearby provinces predict outbreaks here.

Output: neighbor_features.csv
Columns:
  - neighbor_incidence_lag0  : weighted neighbor incidence same week
  - neighbor_incidence_lag1  : weighted neighbor incidence 1 week ago
  - neighbor_incidence_lag2  : weighted neighbor incidence 2 weeks ago
  - n_neighbors              : how many neighbors each province has
────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.dirname(HERE)
ROOT        = os.path.dirname(DATA_DIR)
INPUT_CSV   = os.path.join(ROOT, "input.csv")
CENTROIDS   = os.path.join(DATA_DIR, "province_centroids.csv")
OUTPUT_CSV  = os.path.join(HERE, "neighbor_features.csv")

NEIGHBOR_KM = 150   # provinces within this distance are considered neighbors

# ── Load data ──────────────────────────────────────────────────
print("Loading input.csv ...")
df = pd.read_csv(INPUT_CSV)
df = df.sort_values(["province", "year", "week"]).reset_index(drop=True)

print("Loading province centroids ...")
centroids = pd.read_csv(CENTROIDS)

# Build a lookup: province name prefix (e.g. "01") -> lat/lon
# input.csv province column looks like "01 Distrito Nacional"
centroids["prov_key"] = centroids["province_id"].astype(str).str.zfill(2)
df["prov_key"] = df["province"].str[:2]

# ── Compute incidence per 100k ─────────────────────────────────
df["incidence"] = df["total_cases"] / df["population"] * 100000

# ── Build distance matrix ──────────────────────────────────────
print("Building distance matrix ...")

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

prov_keys = centroids["prov_key"].tolist()
n = len(prov_keys)
dist_matrix = {}

for i, pk_i in enumerate(prov_keys):
    row_i = centroids[centroids["prov_key"] == pk_i].iloc[0]
    neighbors = {}
    for j, pk_j in enumerate(prov_keys):
        if pk_i == pk_j:
            continue
        row_j = centroids[centroids["prov_key"] == pk_j].iloc[0]
        d = haversine_km(row_i["latitude"], row_i["longitude"],
                         row_j["latitude"], row_j["longitude"])
        if d <= NEIGHBOR_KM:
            neighbors[pk_j] = d
    dist_matrix[pk_i] = neighbors

# Print neighbor counts
for pk, nbrs in dist_matrix.items():
    name = centroids[centroids["prov_key"] == pk]["province_name"].values[0]
    print(f"  {name}: {len(nbrs)} neighbors within {NEIGHBOR_KM}km")

# ── Compute weekly incidence pivot ────────────────────────────
print("Computing neighbor incidence features ...")
pivot = df.pivot_table(index=["year", "week"], columns="prov_key",
                       values="incidence", aggfunc="mean")

results = []

for prov_key, neighbors in dist_matrix.items():
    if not neighbors:
        continue

    prov_rows = df[df["prov_key"] == prov_key].copy().reset_index(drop=True)
    province_name = prov_rows["province"].iloc[0]

    nbr_keys  = list(neighbors.keys())
    distances = np.array([neighbors[k] for k in nbr_keys])
    weights   = 1.0 / distances
    weights   = weights / weights.sum()

    lag0_list, lag1_list, lag2_list = [], [], []

    for _, row in prov_rows.iterrows():
        y, w = row["year"], row["week"]

        def get_weighted(yr, wk):
            try:
                vals = pivot.loc[(yr, wk), nbr_keys]
                vals = vals.fillna(0).values.astype(float)
                return float(np.dot(weights, vals))
            except KeyError:
                return np.nan

        # Same week neighbors
        lag0_list.append(get_weighted(y, w))

        # 1 week lag
        w1, y1 = (w - 1, y) if w > 1 else (52, y - 1)
        lag1_list.append(get_weighted(y1, w1))

        # 2 week lag
        w2, y2 = (w - 2, y) if w > 2 else (52 + w - 2, y - 1)
        lag2_list.append(get_weighted(y2, w2))

    prov_rows["neighbor_incidence_lag0"] = lag0_list
    prov_rows["neighbor_incidence_lag1"] = lag1_list
    prov_rows["neighbor_incidence_lag2"] = lag2_list
    prov_rows["n_neighbors"] = len(neighbors)

    results.append(prov_rows[[
        "province", "year", "week",
        "neighbor_incidence_lag0",
        "neighbor_incidence_lag1",
        "neighbor_incidence_lag2",
        "n_neighbors",
    ]])

out = pd.concat(results, ignore_index=True)
out.to_csv(OUTPUT_CSV, index=False)
print(f"Done. Saved {len(out):,} rows to {OUTPUT_CSV}")
