"""
run_island_ga_v2.py
────────────────────────────────────────────────────────────────
Multi-island SINDy with Genetic Algorithm crossover.
Uses the v2 autoencoder checkpoint: best_model_latent6_v2.pt
val_r2 of v2 autoencoder: 0.8819

Architecture: [1612, 2048, 2048, 1024, 128, 6]
(monotone descending, batchnorm, elu, dropout=0.4282)

Run from project root:
  python3 autoencoder-sindy/run_island_ga_v2.py

Outputs in autoencoder-sindy/island_ga_v2/
────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import torch
import torch.nn as nn

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
AE_DIR  = os.path.join(ROOT, "autoencoder")
OUT_DIR = os.path.join(HERE, "island_ga_v2")
os.makedirs(OUT_DIR, exist_ok=True)

VERSION_LABEL = "v2"

# ═══════════════════════════════════════════════════════════════
# AUTOENCODER CONFIG for v2
# ═══════════════════════════════════════════════════════════════
LATENT_DIM      = 6
HIDDEN_DIMS     = [2048, 2048, 1024]   # monotone=True, already sorted descending
PRE_BOTTLENECK  = 128
SHALLOW_DECODER = False
USE_BATCHNORM   = True
DROPOUT         = 0.4282
ACTIVATION      = "elu"

MODEL_PATH = os.path.join(AE_DIR, "best_model_latent6_v2.pt")

# ═══════════════════════════════════════════════════════════════
# ISLAND GA CONFIG
# ═══════════════════════════════════════════════════════════════
N_ISLANDS        = 8
PROVINCE_FRAC    = 0.52
ISLAND_SEED_BASE = 100
N_CROSS_VARIANTS = 20
CROSS_TERM_PROB  = 0.5

# SINDy config
STLSQ_THRESH   = 0.20
POLY_DEG       = 1
WEEKS_PER_YEAR = 52
INCIDENCE_COL  = 28

TRAIN_YEARS = list(range(2015, 2020))
VAL_YEARS   = [2020, 2021]
TEST_YEARS  = [2022, 2023]

EXTERNAL_POOL = [
    "nino34_anom", "neighbor_incidence_lag1",
    "consecutive_wet_weeks", "solar_radiation",
    "soi_index", "avg_rainfall", "avg_temp",
]

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
print(f"Island GA  {VERSION_LABEL}  Loading data ...")
df = pd.read_csv(os.path.join(ROOT, "extended_input_normalized.csv"))
ID_COLS      = ["province", "year", "month", "week"]
FEATURE_COLS = [c for c in df.columns if c not in ID_COLS]
INPUT_DIM    = WEEKS_PER_YEAR * len(FEATURE_COLS)

df = df.sort_values(["province", "year", "week"]).reset_index(drop=True)
chunks, labels = [], []
for (province, year), grp in df.groupby(["province", "year"]):
    grp_s = grp.sort_values("week")
    if len(grp_s) < WEEKS_PER_YEAR:
        continue
    chunks.append(grp_s[FEATURE_COLS].values[:WEEKS_PER_YEAR].flatten())
    labels.append({"province": province, "year": year})

chunks = np.array(chunks, dtype=np.float32)
if np.isnan(chunks).any():
    col_means = np.nanmean(chunks, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    mask = np.isnan(chunks)
    chunks[mask] = np.take(col_means, np.where(mask)[1])

print(f"  Total province-year chunks: {len(chunks)}")

# ═══════════════════════════════════════════════════════════════
# AUTOENCODER
# ═══════════════════════════════════════════════════════════════
print(f"\nLoading {VERSION_LABEL} autoencoder from {MODEL_PATH} ...")

ACTS = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "elu": nn.ELU, "selu": nn.SELU}

class DengueAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, pre_bottleneck,
                 dropout, activation, use_batchnorm, shallow_decoder):
        super().__init__()
        act = ACTS[activation]
        enc_dims = [input_dim] + hidden_dims
        if pre_bottleneck:
            enc_dims += [pre_bottleneck]
        enc_dims += [latent_dim]
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
        if shallow_decoder:
            self.decoder = nn.Sequential(nn.Linear(latent_dim, input_dim), nn.Sigmoid())
        else:
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

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = DengueAutoencoder(INPUT_DIM, LATENT_DIM, HIDDEN_DIMS, PRE_BOTTLENECK,
                          DROPOUT, ACTIVATION, USE_BATCHNORM, SHALLOW_DECODER)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Architecture: [input] + {HIDDEN_DIMS} + [{PRE_BOTTLENECK}] + [{LATENT_DIM}]")

# ═══════════════════════════════════════════════════════════════
# LATENT REPRESENTATIONS
# ═══════════════════════════════════════════════════════════════
print("\nEncoding all province-year chunks ...")
z_cols = [f"z{i+1}" for i in range(LATENT_DIM)]

X_all = torch.tensor(chunks)
with torch.no_grad():
    Z_all = model.encode(X_all).numpy()

latent_df = pd.DataFrame(Z_all, columns=z_cols)
latent_df["province"] = [labels[i]["province"] for i in range(len(labels))]
latent_df["year"]     = [labels[i]["year"]     for i in range(len(labels))]
latent_df["split"]    = [
    "train" if labels[i]["year"] in TRAIN_YEARS
    else ("val" if labels[i]["year"] in VAL_YEARS else "test")
    for i in range(len(labels))
]

# ═══════════════════════════════════════════════════════════════
# SINDY SETUP
# ═══════════════════════════════════════════════════════════════
print("\nSetting up SINDy library ...")

annual = df.groupby(["province", "year"])[FEATURE_COLS].mean().reset_index()
EXTERNAL_POOL_VALID = [c for c in EXTERNAL_POOL if c in annual.columns]
print(f"  External features: {EXTERNAL_POOL_VALID}")

latent_ext = latent_df.merge(
    annual[["province", "year"] + EXTERNAL_POOL_VALID],
    on=["province", "year"], how="left"
)

train_latent_ext = latent_ext[latent_ext["year"].isin(TRAIN_YEARS)]
z_scaler   = StandardScaler().fit(train_latent_ext[z_cols].values)
ext_scaler = StandardScaler().fit(train_latent_ext[EXTERNAL_POOL_VALID].values)

ext_train_min = train_latent_ext[EXTERNAL_POOL_VALID].min().values
ext_train_max = train_latent_ext[EXTERNAL_POOL_VALID].max().values

poly = PolynomialFeatures(degree=POLY_DEG, include_bias=True)
z_scaled_train = z_scaler.transform(train_latent_ext[z_cols].values)
poly.fit(z_scaled_train)
z_term_names = list(poly.get_feature_names_out(z_cols))
ALL_TERMS    = z_term_names + EXTERNAL_POOL_VALID
N_TERMS      = len(ALL_TERMS)
print(f"  Library terms: {N_TERMS}  (poly_deg={POLY_DEG}, z_terms={len(z_term_names)}, ext={len(EXTERNAL_POOL_VALID)})")

def build_theta(rows_df):
    z_s   = z_scaler.transform(rows_df[z_cols].values.astype(float))
    ext_s = ext_scaler.transform(rows_df[EXTERNAL_POOL_VALID].values.astype(float))
    Tz    = poly.transform(z_s)
    return np.hstack([Tz, ext_s])

def build_pairs(df_ext, allowed_start_years, province_subset=None):
    Zrow_list, dZ_list = [], []
    for province, grp in df_ext.groupby("province"):
        if province_subset is not None and province not in province_subset:
            continue
        grp = grp.sort_values("year").reset_index(drop=True)
        for t in range(len(grp) - 1):
            if grp.loc[t, "year"] not in allowed_start_years:
                continue
            if grp.loc[t + 1, "year"] - grp.loc[t, "year"] != 1:
                continue
            Zrow_list.append(grp.loc[t])
            z_t1 = z_scaler.transform(grp.loc[t + 1, z_cols].values.reshape(1, -1)).flatten()
            z_t  = z_scaler.transform(grp.loc[t,     z_cols].values.reshape(1, -1)).flatten()
            dZ_list.append(z_t1 - z_t)
    if not Zrow_list:
        return pd.DataFrame(), np.empty((0, LATENT_DIM))
    return pd.DataFrame(Zrow_list).reset_index(drop=True), np.array(dZ_list)

df_tr_all, dZ_tr_all = build_pairs(latent_ext, TRAIN_YEARS[:-1])
df_va_all, dZ_va_all = build_pairs(latent_ext, [TRAIN_YEARS[-1]] + VAL_YEARS[:-1])
Theta_tr_all = build_theta(df_tr_all)
Theta_va_all = build_theta(df_va_all)
print(f"  All-province train pairs: {len(df_tr_all)}   val pairs: {len(df_va_all)}")

def stlsq(Theta, dZ, threshold, max_iter=30):
    if Theta.shape[0] == 0:
        return np.zeros((Theta.shape[1], dZ.shape[1]))
    Xi = np.linalg.lstsq(Theta, dZ, rcond=None)[0]
    for _ in range(max_iter):
        prev = Xi.copy()
        for j in range(dZ.shape[1]):
            active = np.abs(Xi[:, j]) >= threshold
            Xi[:, j] = 0.0
            if active.sum() > 0:
                Xi[active, j] = np.linalg.lstsq(Theta[:, active], dZ[:, j], rcond=None)[0]
        if np.allclose(Xi, prev, atol=1e-12):
            break
    return Xi

def r2_np(true, pred):
    ss_r = ((true - pred) ** 2).sum()
    ss_t = ((true - true.mean()) ** 2).sum()
    return float(1 - ss_r / (ss_t + 1e-10))

def eval_xi_on_val(Xi_full):
    preds = Theta_va_all @ Xi_full
    r2s   = [r2_np(dZ_va_all[:, j], preds[:, j]) for j in range(LATENT_DIM)]
    return r2s, float(np.mean(r2s))

# ═══════════════════════════════════════════════════════════════
# ISLAND DEFINITIONS
# ═══════════════════════════════════════════════════════════════
provinces_all = sorted(latent_ext["province"].unique())
N_PROVINCES   = len(provinces_all)
K_PER_ISLAND  = max(4, int(N_PROVINCES * PROVINCE_FRAC))
print(f"\nProvinces: {N_PROVINCES}   Per island: {K_PER_ISLAND}")

island_subsets = []
for isl in range(N_ISLANDS):
    rng_isl = np.random.default_rng(ISLAND_SEED_BASE + isl)
    chosen  = list(rng_isl.choice(provinces_all, size=K_PER_ISLAND, replace=False))
    island_subsets.append(chosen)

# ═══════════════════════════════════════════════════════════════
# ISLAND TRAINING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"  ISLAND SINDY TRAINING  ({VERSION_LABEL})")
print("=" * 70)

island_results = []

for isl in range(N_ISLANDS):
    subset = island_subsets[isl]
    df_tr_isl, dZ_tr_isl = build_pairs(latent_ext, TRAIN_YEARS[:-1], subset)

    if len(df_tr_isl) < N_TERMS:
        print(f"\n  Island {isl+1}  SKIPPED (only {len(df_tr_isl)} pairs for {N_TERMS} terms)")
        island_results.append(None)
        continue

    Theta_tr_isl = build_theta(df_tr_isl)
    Xi_isl       = stlsq(Theta_tr_isl, dZ_tr_isl, STLSQ_THRESH)

    val_r2s, val_r2_mean = eval_xi_on_val(Xi_isl)
    preds_tr = Theta_tr_isl @ Xi_isl
    tr_r2s   = [r2_np(dZ_tr_isl[:, j], preds_tr[:, j]) for j in range(LATENT_DIM)]

    n_active = int((np.abs(Xi_isl) > 1e-12).any(axis=1).sum())
    gap_pen  = max(0.0, float(np.mean(tr_r2s)) - val_r2_mean - 0.15)
    fitness  = val_r2_mean - 0.01 * n_active - 0.5 * gap_pen

    island_results.append({
        "island":        isl + 1,
        "n_pairs_train": len(df_tr_isl),
        "n_active":      n_active,
        "tr_r2s":        tr_r2s,
        "val_r2s":       val_r2s,
        "val_r2_mean":   val_r2_mean,
        "fitness":       fitness,
        "Xi":            Xi_isl,
        "provinces":     subset,
    })

    print(f"\n  Island {isl+1}/{N_ISLANDS}  pairs={len(df_tr_isl)}  n_active={n_active}")
    print(f"    train_r2s  = {[round(v,3) for v in tr_r2s]}")
    print(f"    val_r2s    = {[round(v,3) for v in val_r2s]}  mean={val_r2_mean:.4f}")
    print(f"    fitness    = {fitness:.4f}")

# ═══════════════════════════════════════════════════════════════
# SELECT TOP 2 ISLANDS
# ═══════════════════════════════════════════════════════════════
valid_results = [r for r in island_results if r is not None]
if len(valid_results) < 2:
    raise RuntimeError("Fewer than 2 valid islands. Reduce N_TERMS or increase PROVINCE_FRAC.")

valid_results.sort(key=lambda x: x["fitness"], reverse=True)
parent_a = valid_results[0]
parent_b = valid_results[1]

print("\n" + "=" * 70)
print("  TOP 2 ISLANDS SELECTED FOR CROSSOVER")
print("=" * 70)
print(f"  Parent A: Island {parent_a['island']}  fitness={parent_a['fitness']:.4f}"
      f"  val_r2_mean={parent_a['val_r2_mean']:.4f}")
print(f"  Parent B: Island {parent_b['island']}  fitness={parent_b['fitness']:.4f}"
      f"  val_r2_mean={parent_b['val_r2_mean']:.4f}")

island_rows = []
for r in valid_results:
    island_rows.append({
        "island":        r["island"],
        "fitness":       round(r["fitness"], 4),
        "val_r2_mean":   round(r["val_r2_mean"], 4),
        "val_r2s":       str([round(v, 3) for v in r["val_r2s"]]),
        "tr_r2s":        str([round(v, 3) for v in r["tr_r2s"]]),
        "n_active":      r["n_active"],
        "n_pairs_train": r["n_pairs_train"],
        "provinces":     str(r["provinces"]),
    })
pd.DataFrame(island_rows).to_csv(os.path.join(OUT_DIR, "island_scores.csv"), index=False)
print(f"\n  Saved island_scores.csv")

# ═══════════════════════════════════════════════════════════════
# GA CROSSOVER
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  GA CROSSOVER")
print("=" * 70)

Xi_a = parent_a["Xi"]
Xi_b = parent_b["Xi"]

active_a = (np.abs(Xi_a) > 1e-12).any(axis=1)
active_b = (np.abs(Xi_b) > 1e-12).any(axis=1)
only_a   = active_a & ~active_b
only_b   = ~active_a & active_b
in_both  = active_a & active_b

print(f"\n  Terms only in A:   {only_a.sum()}")
print(f"  Terms only in B:   {only_b.sum()}")
print(f"  Terms in both:     {in_both.sum()}")
print(f"  Union total:       {(active_a | active_b).sum()}")

def refit_xi(mask):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.zeros((N_TERMS, LATENT_DIM))
    Xi_sub  = stlsq(Theta_tr_all[:, idx], dZ_tr_all, STLSQ_THRESH)
    Xi_full = np.zeros((N_TERMS, LATENT_DIM))
    Xi_full[idx] = Xi_sub
    return Xi_full

union_mask = active_a | active_b
Xi_union   = refit_xi(union_mask)
val_r2s_union, val_mean_union = eval_xi_on_val(Xi_union)
print(f"\n  Union crossover  n_active={(np.abs(Xi_union)>1e-12).any(axis=1).sum()}"
      f"  val_r2_mean={val_mean_union:.4f}"
      f"  val_r2s={[round(v,3) for v in val_r2s_union]}")

rng = np.random.default_rng(77)
best_cross_score = val_mean_union
best_cross_Xi    = Xi_union
best_cross_r2s   = val_r2s_union
best_cross_label = "union"

variant_records = [{"variant": "union",
                    "val_r2_mean": round(val_mean_union, 4),
                    "val_r2s": str([round(v, 3) for v in val_r2s_union]),
                    "n_active": int((np.abs(Xi_union)>1e-12).any(axis=1).sum())}]

for v in range(N_CROSS_VARIANTS):
    cross_mask = in_both.copy()
    for ti in np.where(only_a)[0]:
        if rng.random() < CROSS_TERM_PROB:
            cross_mask[ti] = True
    for ti in np.where(only_b)[0]:
        if rng.random() < CROSS_TERM_PROB:
            cross_mask[ti] = True
    if cross_mask.sum() == 0:
        continue
    Xi_v = refit_xi(cross_mask)
    r2s_v, mean_v = eval_xi_on_val(Xi_v)
    n_act_v = int((np.abs(Xi_v) > 1e-12).any(axis=1).sum())
    variant_records.append({"variant": f"rand_{v+1}",
                             "val_r2_mean": round(mean_v, 4),
                             "val_r2s": str([round(r, 3) for r in r2s_v]),
                             "n_active": n_act_v})
    if mean_v > best_cross_score:
        best_cross_score = mean_v
        best_cross_Xi    = Xi_v
        best_cross_r2s   = r2s_v
        best_cross_label = f"rand_{v+1}"
    if (v + 1) % 5 == 0:
        print(f"  Variant {v+1:3d}/{N_CROSS_VARIANTS}  "
              f"val_r2_mean={mean_v:.4f}  best_so_far={best_cross_score:.4f}")

Xi_cross = best_cross_Xi
print(f"\n  Best crossover: {best_cross_label}"
      f"  val_r2_mean={best_cross_score:.4f}"
      f"  val_r2s={[round(v,3) for v in best_cross_r2s]}")
print(f"  vs Parent A val_r2_mean={parent_a['val_r2_mean']:.4f}")
print(f"  vs Parent B val_r2_mean={parent_b['val_r2_mean']:.4f}")

pd.DataFrame(variant_records).sort_values("val_r2_mean", ascending=False)\
    .to_csv(os.path.join(OUT_DIR, "crossover_variants.csv"), index=False)

n_active_cross = int((np.abs(Xi_cross) > 1e-12).any(axis=1).sum())
coef_df = pd.DataFrame(Xi_cross, index=ALL_TERMS,
                        columns=[f"d{z}/dt" for z in z_cols])
coef_df.to_csv(os.path.join(OUT_DIR, "crossed_coefficients.csv"))
print(f"\n  Saved crossed_coefficients.csv  (n_active={n_active_cross})")
print("\n  Crossed equation terms:")
for i, term in enumerate(ALL_TERMS):
    row = Xi_cross[i]
    if (np.abs(row) > 1e-12).any():
        coef_str = "  ".join(f"d{z}/dt={row[j]:+.5f}" for j, z in enumerate(z_cols)
                              if abs(row[j]) > 1e-12)
        tag = "[EXT]" if term in EXTERNAL_POOL_VALID else "     "
        print(f"    {tag}  {term:30s}  {coef_str}")

# ═══════════════════════════════════════════════════════════════
# STAGE 3 TEST EVALUATION (test data unlocked here only)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STAGE 3  TEST EVALUATION  (2022-2023 unlocked)")
print("=" * 70)

def get_incidence_curve(province, year):
    rows = df[(df["province"] == province) & (df["year"] == year)]
    rows = rows.sort_values("week")
    if len(rows) < WEEKS_PER_YEAR:
        return None
    return rows[FEATURE_COLS[INCIDENCE_COL]].values[:WEEKS_PER_YEAR].astype(float)

actual_incidence_col = FEATURE_COLS[INCIDENCE_COL]
print(f"\n  Incidence column: '{actual_incidence_col}' (index {INCIDENCE_COL})")

test_2022_ext = latent_ext[latent_ext["year"] == 2022][EXTERNAL_POOL_VALID]
if len(test_2022_ext) > 0:
    print("\n  OOD clip diagnostics (2022 mean vs training range):")
    for fi, feat in enumerate(EXTERNAL_POOL_VALID):
        v    = test_2022_ext[feat].mean()
        lo   = ext_train_min[fi]
        hi   = ext_train_max[fi]
        flag = "  <CLIPPED>" if (v < lo or v > hi) else ""
        print(f"    {feat:32s}  val={v:+.4f}  train=[{lo:+.4f}, {hi:+.4f}]{flag}")

eval_records = []

for province in provinces_all:
    sub      = latent_ext[latent_ext["province"] == province].sort_values("year")
    seed_row = sub[sub["year"] == 2021]
    if seed_row.empty:
        continue
    z_cur = seed_row[z_cols].values[0].astype(float)

    for test_year in TEST_YEARS:
        test_row = sub[sub["year"] == test_year]
        if test_row.empty:
            continue

        row_df   = pd.DataFrame([test_row.iloc[0]])
        row_df_z = row_df.copy()
        for k, zc in enumerate(z_cols):
            row_df_z[zc] = z_cur[k]
        for fi, feat in enumerate(EXTERNAL_POOL_VALID):
            row_df_z[feat] = float(np.clip(
                float(row_df_z[feat].values[0]),
                ext_train_min[fi], ext_train_max[fi]
            ))

        t_row     = build_theta(row_df_z)
        dz_scaled = (t_row @ Xi_cross).flatten()
        dz_actual = dz_scaled * z_scaler.scale_
        z_pred    = z_cur + dz_actual
        z_cur     = z_pred.copy()

        with torch.no_grad():
            z_tensor = torch.tensor(z_pred.astype(np.float32)).unsqueeze(0)
            decoded  = model.decode(z_tensor).numpy().flatten()

        decoded_2d     = decoded.reshape(WEEKS_PER_YEAR, len(FEATURE_COLS))
        pred_incidence = decoded_2d[:, INCIDENCE_COL]
        actual_incidence = get_incidence_curve(province, test_year)
        if actual_incidence is None:
            continue

        pred_didt   = np.diff(pred_incidence)
        actual_didt = np.diff(actual_incidence)

        eval_records.append({
            "province":         province,
            "year":             test_year,
            "r2_incidence":     r2_np(actual_incidence, pred_incidence),
            "mae_incidence":    float(np.abs(actual_incidence - pred_incidence).mean()),
            "r2_didt":          r2_np(actual_didt, pred_didt),
            "mae_didt":         float(np.abs(actual_didt - pred_didt).mean()),
            "pred_incidence":   pred_incidence.tolist(),
            "actual_incidence": actual_incidence.tolist(),
            "pred_didt":        pred_didt.tolist(),
            "actual_didt":      actual_didt.tolist(),
            "z_pred":           z_pred.tolist(),
            "z_actual":         test_row[z_cols].values[0].tolist(),
        })

eval_df = pd.DataFrame(eval_records)

print("\n  Results by province (mean across test years):")
prov_summary = eval_df.groupby("province")[["r2_incidence", "r2_didt"]].mean()\
    .sort_values("r2_incidence", ascending=False)
print(prov_summary.round(3).to_string())

print(f"\n  Summary across all test provinces ({VERSION_LABEL}):")
print(f"  Incidence R2  mean={eval_df['r2_incidence'].mean():.4f}"
      f"  median={eval_df['r2_incidence'].median():.4f}"
      f"  min={eval_df['r2_incidence'].min():.4f}"
      f"  max={eval_df['r2_incidence'].max():.4f}")
print(f"  dI/dt     R2  mean={eval_df['r2_didt'].mean():.4f}"
      f"  median={eval_df['r2_didt'].median():.4f}")

metrics_save = eval_df.drop(columns=["pred_incidence","actual_incidence",
                                      "pred_didt","actual_didt",
                                      "z_pred","z_actual"])
metrics_save.to_csv(os.path.join(OUT_DIR, "eval_metrics.csv"), index=False)
print(f"\n  Saved eval_metrics.csv")

# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════
print("\nGenerating plots ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
island_nums   = [r["island"]      for r in valid_results]
island_scores = [r["val_r2_mean"] for r in valid_results]
island_fit    = [r["fitness"]     for r in valid_results]
colors = ["#e63946" if r["island"] in [parent_a["island"], parent_b["island"]]
          else "#457b9d" for r in valid_results]
axes[0].bar(island_nums, island_scores, color=colors)
axes[0].set_xlabel("Island")
axes[0].set_ylabel("Val R2 mean")
axes[0].set_title(f"Island Val R2  {VERSION_LABEL}  (red = crossover parents)")
axes[0].set_xticks(island_nums)
axes[0].grid(axis="y", alpha=0.3)
axes[1].bar(island_nums, island_fit, color=colors)
axes[1].set_xlabel("Island")
axes[1].set_ylabel("Fitness score")
axes[1].set_title(f"Island Fitness  {VERSION_LABEL}")
axes[1].set_xticks(island_nums)
axes[1].grid(axis="y", alpha=0.3)
plt.suptitle(f"Island SINDy GA  {VERSION_LABEL}  (dim={LATENT_DIM}  N_islands={N_ISLANDS})", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_island_scores.png"), dpi=150)
plt.close()

sample_provs = sorted(provinces_all)[:8]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
weeks = np.arange(1, WEEKS_PER_YEAR + 1)
for pi, province in enumerate(sample_provs):
    ax  = axes[pi]
    sub = eval_df[eval_df["province"] == province].sort_values("year")
    colors_yr = {2022: ("#457b9d", "#e63946"), 2023: ("#2a9d8f", "#f4a261")}
    for _, row in sub.iterrows():
        yr = row["year"]
        ac, pc = colors_yr.get(yr, ("#888", "#ccc"))
        ax.plot(weeks, row["actual_incidence"], "-",  color=ac, lw=2, alpha=0.9, label=f"{yr} actual")
        ax.plot(weeks, row["pred_incidence"],   "--", color=pc, lw=2, alpha=0.9, label=f"{yr} pred")
    mean_r2 = sub["r2_incidence"].mean()
    short = province.split(" ", 1)[1][:14] if " " in province else province[:14]
    ax.set_title(f"{short}  R2={mean_r2:.2f}", fontsize=9)
    ax.set_xlabel("Week", fontsize=8)
    ax.set_ylabel("Incidence (norm.)", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)
    if pi == 0:
        ax.legend(fontsize=6, ncol=2)
plt.suptitle(
    f"Island GA {VERSION_LABEL}  Predicted vs Actual  (Test 2022-2023)\n"
    f"Incidence R2  mean={eval_df['r2_incidence'].mean():.4f}  "
    f"median={eval_df['r2_incidence'].median():.4f}", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_incidence.png"), dpi=150)
plt.close()

prov_r2 = eval_df.groupby("province")[["r2_incidence", "r2_didt"]].mean()
fig, ax = plt.subplots(figsize=(6, max(7, len(prov_r2) * 0.32)))
im = ax.imshow(prov_r2.values, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_xticks([0, 1])
ax.set_xticklabels(["R2 incidence", "R2 dI/dt"])
ax.set_yticks(range(len(prov_r2)))
ax.set_yticklabels(
    [p.split(" ", 1)[1][:22] if " " in p else p[:22] for p in prov_r2.index], fontsize=7)
for i in range(len(prov_r2)):
    for j in range(2):
        ax.text(j, i, f"{prov_r2.values[i,j]:.2f}", ha="center", va="center",
                fontsize=6.5, color="black")
plt.colorbar(im, ax=ax, label="R2")
ax.set_title(f"Island GA {VERSION_LABEL}  Test R2 per Province\n(mean 2022+2023)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_r2_heatmap.png"), dpi=150)
plt.close()

print("  Saved plot_island_scores.png")
print("  Saved plot_incidence.png")
print("  Saved plot_r2_heatmap.png")

print("\n" + "=" * 70)
print(f"  ISLAND GA {VERSION_LABEL}  COMPLETE")
print("=" * 70)
print(f"  AE val_r2 = 0.8819  (v2 checkpoint)")
print(f"  Incidence R2  mean={eval_df['r2_incidence'].mean():.4f}"
      f"  median={eval_df['r2_incidence'].median():.4f}")
print(f"  dI/dt     R2  mean={eval_df['r2_didt'].mean():.4f}")
print(f"  Crossed equation uses {n_active_cross} active terms")
print(f"  Best crossover variant: {best_cross_label}")
print(f"\n  Outputs in: autoencoder-sindy/island_ga_v2/")
print("─" * 70)
