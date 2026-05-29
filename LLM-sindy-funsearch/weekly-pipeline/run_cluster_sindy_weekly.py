"""
run_cluster_sindy_weekly.py
────────────────────────────────────────────────────────────────
SINDy per province cluster using weekly latent representations.

Why weekly and not annual:
  Annual latent gives 5 training pairs per province = ~15-55 per
  cluster, far too few for STLSQ. Weekly latent gives 260 pairs
  per province = 780-2860 per cluster. Terms-to-pairs ratio goes
  from catastrophic (<5) to healthy (60-220).

Pipeline:
  1. Load province_clusters.csv  (5 clusters from K-means)
  2. Load latent_weekly_dimK.csv (train+val rows)
  3. For each cluster:
     a. Gather consecutive (z_t, z_{t+1}) pairs WITHIN each province
     b. Fit STLSQ on train pairs: dZ = Xi(Z) * coefficients
     c. Evaluate rollout R2 on val provinces
  4. Print cluster summary. Unlock test at end.

CONFIG:
  LATENT_DIM   Change to 3 or 4 if 6 is too wide for small clusters.
  SINDY_DIMS   Indices of z dims included in SINDy (drop noisy dims).
  POLY_DEG     1 = linear library (13 terms for 6 dims).
                2 = quadratic (35 terms, needs large clusters).
  STLSQ_THRESH Sparsity threshold. Raise to get sparser equations.

Outputs (in weekly-pipeline/):
  cluster_sindy_equations.txt      discovered equations per cluster
  cluster_sindy_results.csv        cluster-level val and test R2
  cluster_sindy_byregion.csv       per-province val and test R2
────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, "extended_input_normalized.csv")

LATENT_DIM   = 6
LATENT_CSV   = os.path.join(HERE, "latents", f"latent_weekly_dim{LATENT_DIM}.csv")
MODEL_PATH   = os.path.join(HERE, "models", f"best_model_weekly_dim{LATENT_DIM}.pt")
CLUSTER_CSV  = os.path.join(HERE, "province_clusters.csv")

# ── SINDy CONFIG ───────────────────────────────────────────────
SINDY_DIMS   = list(range(LATENT_DIM))   # use all dims by default
POLY_DEG     = 1
STLSQ_THRESH = 0.20    # raised from 0.05 to force sparser equations
STLSQ_ITERS  = 20

TRAIN_YEARS = list(range(2015, 2020))
VAL_YEARS   = [2020, 2021]
TEST_YEARS  = [2022, 2023]

N_SINDY = len(SINDY_DIMS)
print(f"Cluster SINDy  latent_dim={LATENT_DIM}  sindy_dims={SINDY_DIMS}")
print(f"  poly_deg={POLY_DEG}  stlsq_thresh={STLSQ_THRESH}\n")

# ── Encode test rows through the saved weekly autoencoder ──────
ACTS = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "elu": nn.ELU}

class WeeklyAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims,
                 dropout, activation, use_batchnorm):
        super().__init__()
        act = ACTS[activation]
        enc_dims = [input_dim] + hidden_dims + [latent_dim]
        enc_layers = []
        for i in range(len(enc_dims) - 1):
            enc_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i < len(enc_dims) - 2:
                if use_batchnorm:
                    enc_layers.append(nn.BatchNorm1d(enc_dims[i + 1]))
                enc_layers.append(act())
                if dropout > 0.0:
                    enc_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*enc_layers)
        dec_dims = list(reversed(enc_dims))
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                if use_batchnorm:
                    dec_layers.append(nn.BatchNorm1d(dec_dims[i + 1]))
                dec_layers.append(act())
                if dropout > 0.0:
                    dec_layers.append(nn.Dropout(dropout))
            else:
                dec_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x): return self.decoder(self.encoder(x))
    def encode(self, x):  return self.encoder(x)

raw = pd.read_csv(DATA_CSV)
raw = raw.sort_values(["province", "year", "week"]).reset_index(drop=True)
ID_COLS      = ["province", "year", "month", "week"]
FEATURE_COLS = [c for c in raw.columns if c not in ID_COLS]
raw["time_sin"] = np.sin(2.0 * np.pi * raw["week"] / 52.0)
raw["time_cos"] = np.cos(2.0 * np.pi * raw["week"] / 52.0)
INPUT_FEATURES = FEATURE_COLS + ["time_sin", "time_cos"]
INPUT_DIM = len(INPUT_FEATURES)

train_mask = raw["year"].isin(TRAIN_YEARS)
col_means  = raw.loc[train_mask, INPUT_FEATURES].mean()
raw[INPUT_FEATURES] = raw[INPUT_FEATURES].fillna(col_means)

ae = WeeklyAutoencoder(INPUT_DIM, LATENT_DIM, [512, 256], 0.158, "leakyrelu", True)
ae.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
ae.eval()

test_mask = raw["year"].isin(TEST_YEARS)
X_test_raw = torch.tensor(raw.loc[test_mask, INPUT_FEATURES].values.astype(np.float32))
with torch.no_grad():
    Z_test = ae.encode(X_test_raw).numpy()

z_cols_all = [f"z{i+1}" for i in range(LATENT_DIM)]
test_latent = pd.DataFrame(Z_test, columns=z_cols_all)
test_latent["province"] = raw.loc[test_mask, "province"].values
test_latent["year"]     = raw.loc[test_mask, "year"].values
test_latent["week"]     = raw.loc[test_mask, "week"].values
test_latent["split"]    = "test"

# ── Load data ──────────────────────────────────────────────────
saved_latent = pd.read_csv(LATENT_CSV)
latent = pd.concat([saved_latent, test_latent], ignore_index=True)

clusters = pd.read_csv(CLUSTER_CSV)[["province", "cluster_id"]]
latent   = latent.merge(clusters, on="province", how="left")

z_cols_sindy = [z_cols_all[i] for i in SINDY_DIMS]

# Load raw incidence for evaluation target
raw = pd.read_csv(DATA_CSV)
raw = raw.sort_values(["province", "year", "week"]).reset_index(drop=True)

# ── SINDy helpers ──────────────────────────────────────────────
poly = PolynomialFeatures(degree=POLY_DEG, include_bias=True)

def build_library(Z):
    """Z shape (n, n_sindy_dims) -> library shape (n, n_terms)."""
    return poly.fit_transform(Z)

def get_feature_names():
    dummy = np.zeros((1, N_SINDY))
    poly.fit_transform(dummy)
    names = poly.get_feature_names_out([f"z{SINDY_DIMS[i]+1}" for i in range(N_SINDY)])
    return list(names)

def stlsq(Theta, dZ, threshold, n_iters):
    """
    Sequentially-Thresholded Least Squares.
    Theta: (n, n_terms)  dZ: (n, n_sindy_dims)
    Returns Xi: (n_terms, n_sindy_dims)
    """
    Xi = np.linalg.lstsq(Theta, dZ, rcond=None)[0]
    for _ in range(n_iters):
        small = np.abs(Xi) < threshold
        Xi[small] = 0.0
        for k in range(dZ.shape[1]):
            active = ~small[:, k]
            if active.sum() == 0:
                continue
            Xi[active, k] = np.linalg.lstsq(
                Theta[:, active], dZ[:, k], rcond=None)[0]
    return Xi

def build_pairs(province_df, year_list, z_scaler):
    """
    Build consecutive (z_t, dz_t) pairs from weekly latent data.
    Pairs are only taken WITHIN the same province and same year,
    preventing boundary jumps between provinces or year ends.
    """
    Z_list, dZ_list = [], []
    for (prov, yr), grp in province_df[province_df["year"].isin(year_list)].groupby(
            ["province", "year"]):
        grp = grp.sort_values("week").reset_index(drop=True)
        if len(grp) < 2:
            continue
        Z_raw = grp[z_cols_all].values
        Z_sc  = z_scaler.transform(Z_raw)
        Z_s   = Z_sc[:, SINDY_DIMS]
        for t in range(len(Z_s) - 1):
            Z_list.append(Z_s[t])
            dZ_list.append(Z_s[t + 1] - Z_s[t])
    if not Z_list:
        return np.zeros((0, N_SINDY)), np.zeros((0, N_SINDY))
    return np.array(Z_list), np.array(dZ_list)

def rollout_r2(province_df, year_list, Xi, z_scaler, prov_name=None):
    """
    Roll out the discovered equation and compute R2 against
    the true latent trajectory for each province in year_list.
    Returns list of (province, year, r2) tuples.
    """
    results = []
    sel = province_df[province_df["year"].isin(year_list)]
    if prov_name:
        sel = sel[sel["province"] == prov_name]

    for (prov, yr), grp in sel.groupby(["province", "year"]):
        grp = grp.sort_values("week").reset_index(drop=True)
        if len(grp) < 2:
            continue
        Z_raw = grp[z_cols_all].values
        Z_sc  = z_scaler.transform(Z_raw)

        z_pred = Z_sc[0].copy()
        preds  = [z_pred.copy()]
        for _ in range(len(Z_sc) - 1):
            z_cur  = z_pred.copy()
            z_sindy = z_cur[SINDY_DIMS].reshape(1, -1)
            theta   = poly.transform(z_sindy)
            dz      = (theta @ Xi).flatten()
            z_pred[SINDY_DIMS] = z_cur[SINDY_DIMS] + dz
            preds.append(z_pred.copy())

        Z_pred_sc = np.array(preds)
        Z_true_sc = Z_sc

        # R2 over sindy dims only
        r2 = r2_score(Z_true_sc[:, SINDY_DIMS],
                      Z_pred_sc[:, SINDY_DIMS])
        results.append({"province": prov, "year": yr, "r2": round(r2, 4)})
    return results

# ── Main loop: one SINDy per cluster ─────────────────────────
cluster_ids   = sorted(latent["cluster_id"].unique())
all_results   = []
all_byregion  = []
equation_lines = []

for cid in cluster_ids:
    c_df     = latent[latent["cluster_id"] == cid]
    c_provs  = sorted(c_df["province"].unique())
    n_provs  = len(c_provs)

    print("=" * 60)
    print(f"  Cluster {cid}  ({n_provs} provinces)")
    for p in c_provs:
        print(f"    {p}")

    # Fit z scaler on train data of this cluster
    train_rows = c_df[c_df["year"].isin(TRAIN_YEARS)][z_cols_all].values
    z_scaler   = StandardScaler()
    z_scaler.fit(train_rows)

    # Build train pairs
    Z_tr, dZ_tr = build_pairs(c_df, TRAIN_YEARS, z_scaler)
    Z_va, dZ_va = build_pairs(c_df, VAL_YEARS,   z_scaler)

    n_terms = build_library(np.zeros((1, N_SINDY))).shape[1]
    ratio   = len(Z_tr) / max(n_terms, 1)
    print(f"\n  Train pairs: {len(Z_tr)}  Terms: {n_terms}  Ratio: {ratio:.1f}")

    if len(Z_tr) < n_terms:
        print(f"  SKIP: too few training pairs for {n_terms} terms.")
        continue

    # Fit SINDy
    poly.fit(np.zeros((1, N_SINDY)))   # initialise feature names
    Theta_tr = build_library(Z_tr)
    Xi = stlsq(Theta_tr, dZ_tr, STLSQ_THRESH, STLSQ_ITERS)

    n_active = int((Xi != 0).sum())
    feat_names = get_feature_names()
    print(f"  Active terms: {n_active} / {n_terms * N_SINDY}")

    # Val rollout
    val_rows = rollout_r2(c_df, VAL_YEARS, Xi, z_scaler)
    if val_rows:
        val_r2_mean   = float(np.mean([r["r2"] for r in val_rows]))
        val_r2_median = float(np.median([r["r2"] for r in val_rows]))
    else:
        val_r2_mean = val_r2_median = float("nan")

    print(f"  Val  mean_r2={val_r2_mean:+.4f}  median_r2={val_r2_median:+.4f}")

    # Test rollout (LOCKED until here)
    Z_te, dZ_te = build_pairs(c_df, TEST_YEARS, z_scaler)
    test_rows = rollout_r2(c_df, TEST_YEARS, Xi, z_scaler)
    if test_rows:
        test_r2_mean   = float(np.mean([r["r2"] for r in test_rows]))
        test_r2_median = float(np.median([r["r2"] for r in test_rows]))
    else:
        test_r2_mean = test_r2_median = float("nan")

    print(f"  Test mean_r2={test_r2_mean:+.4f}  median_r2={test_r2_median:+.4f}")

    all_results.append({
        "cluster_id"     : cid,
        "n_provinces"    : n_provs,
        "train_pairs"    : len(Z_tr),
        "active_terms"   : n_active,
        "val_r2_mean"    : round(val_r2_mean, 4),
        "val_r2_median"  : round(val_r2_median, 4),
        "test_r2_mean"   : round(test_r2_mean, 4),
        "test_r2_median" : round(test_r2_median, 4),
    })

    for row in val_rows:
        row["cluster_id"] = cid
        row["split"] = "val"
        all_byregion.append(row)
    for row in test_rows:
        row["cluster_id"] = cid
        row["split"] = "test"
        all_byregion.append(row)

    # Format equations
    equation_lines.append(f"\nCluster {cid}  ({n_provs} provinces: {', '.join(c_provs)})\n")
    equation_lines.append(f"  Val mean_r2={val_r2_mean:.4f}  Test mean_r2={test_r2_mean:.4f}\n")
    for k, zdim in enumerate(SINDY_DIMS):
        terms = [(feat_names[j], Xi[j, k])
                 for j in range(len(feat_names)) if Xi[j, k] != 0]
        eq_str = " + ".join(f"{v:.4f}*{n}" for n, v in terms) if terms else "0"
        equation_lines.append(f"  dz{zdim+1}/dt = {eq_str}\n")
    equation_lines.append("\n")
    print()

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CLUSTER SINDY SUMMARY")
print("=" * 60)
res_df = pd.DataFrame(all_results)
print(res_df.to_string(index=False))
res_df.to_csv(os.path.join(HERE, "cluster_sindy_results.csv"), index=False)

region_df = pd.DataFrame(all_byregion)
region_df.to_csv(os.path.join(HERE, "cluster_sindy_byregion.csv"), index=False)

eq_path = os.path.join(HERE, "cluster_sindy_equations.txt")
with open(eq_path, "w") as f:
    f.write(f"CLUSTER SINDY EQUATIONS  latent_dim={LATENT_DIM}  poly_deg={POLY_DEG}  thresh={STLSQ_THRESH}\n")
    f.writelines(equation_lines)

print(f"\n  Equations saved to cluster_sindy_equations.txt")
print(f"  Results saved to cluster_sindy_results.csv")
print(f"  Per-province saved to cluster_sindy_byregion.csv")

if not res_df.empty:
    overall_val  = res_df["val_r2_mean"].mean()
    overall_test = res_df["test_r2_mean"].mean()
    print(f"\n  Overall mean val_r2  = {overall_val:.4f}")
    print(f"  Overall mean test_r2 = {overall_test:.4f}")
    print()
    if overall_test < 0.1:
        print("  NOTE: if results are modest, re-run with LATENT_DIM=3 or 4")
        print("  and SINDY_DIMS = list(range(LATENT_DIM)) to reduce library size.")
print("─" * 60)
