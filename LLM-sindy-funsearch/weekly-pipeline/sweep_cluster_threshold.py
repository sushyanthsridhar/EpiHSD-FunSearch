"""
sweep_cluster_threshold.py
────────────────────────────────────────────────────────────
Sweep THRESH_CLUSTER over a range of values and report the
province-mean val and test incidence R2 for each.

Fits W_global once, then for each threshold candidate refits
only the cluster and province equations. Fast: roughly 10-15
seconds per threshold value.

Sweep covers THRESH_CLUSTER from 0.05 to 0.40. THRESH_INC_CLUSTER
is swept jointly (same value) since the incidence output is the
metric we care about.

Run:
    python sweep_cluster_threshold.py
────────────────────────────────────────────────────────────
"""

import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Any
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, "extended_input_normalized.csv")

LATENT_DIM  = 6
LATENT_CSV  = os.path.join(HERE, "latents", f"latent_weekly_dim{LATENT_DIM}.csv")
MODEL_PATH  = os.path.join(HERE, "models", f"best_model_weekly_dim{LATENT_DIM}.pt")
CLUSTER_CSV = os.path.join(HERE, "province_clusters.csv")
BEST_JSON   = os.path.join(HERE, "funsearch_best.json")

# Fixed settings
POLY_DEG            = 1
THRESH_GLOBAL       = 0.05
THRESH_PROVINCE     = 0.08
THRESH_INC_GLOBAL   = 0.04
THRESH_INC_PROVINCE = 100.0
LOG_INCIDENCE       = True
STLSQ_ITERS         = 30

TRAIN_YEARS = list(range(2015, 2020))
VAL_YEARS   = [2020, 2021]
TEST_YEARS  = [2022, 2023]

# Thresholds to sweep
CLUSTER_THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40]

# ════════════════════════════════════════════════════════════
# PROGRAM DATACLASSES
# ════════════════════════════════════════════════════════════

@dataclass
class ExtraTerm:
    term_type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def compute(self, X_lag, n_lags, state_dim, inc_col,
                weeks=None):
        try:
            tt, p, ml, sd = self.term_type, self.params, n_lags, state_dim
            if tt == "inc_pow":
                k = min(p.get("lag", 0), ml)
                v = X_lag[:, sd * k + inc_col] ** p.get("power", 2)
            elif tt == "inc_z_prod":
                ki = min(p.get("lag_inc", 0), ml)
                kz = min(p.get("lag_z",   0), ml)
                j  = min(p.get("z_idx",   0), LATENT_DIM - 1)
                v  = X_lag[:, sd * ki + inc_col] * X_lag[:, sd * kz + j]
            elif tt == "cross_inc":
                k1 = min(p.get("lag1", 0), ml)
                k2 = min(p.get("lag2", 1), ml)
                v  = X_lag[:, sd * k1 + inc_col] * X_lag[:, sd * k2 + inc_col]
            elif tt == "z_sq":
                k = min(p.get("lag", 0), ml)
                j = min(p.get("z_idx", 0), LATENT_DIM - 1)
                v = X_lag[:, sd * k + j] ** 2
            elif tt == "z_prod":
                k = min(p.get("lag", 0), ml)
                i = min(p.get("z_idx1", 0), LATENT_DIM - 1)
                j = min(p.get("z_idx2", 1), LATENT_DIM - 1)
                v = X_lag[:, sd * k + i] * X_lag[:, sd * k + j]
            elif tt == "inc_diff":
                k1 = min(p.get("lag1", 0), ml)
                k2 = min(p.get("lag2", 1), ml)
                v  = (X_lag[:, sd * k1 + inc_col]
                      - X_lag[:, sd * k2 + inc_col])
            elif tt == "z_cube":
                k = min(p.get("lag", 0), ml)
                j = min(p.get("z_idx", 0), LATENT_DIM - 1)
                v = X_lag[:, sd * k + j] ** 3
            elif tt == "seasonal":
                if weeks is None:
                    return None
                angle = 2.0 * np.pi * weeks / 52.0
                v = np.cos(angle) if p.get("use_cos", 0) else np.sin(angle)
            else:
                return None
            return v.astype(np.float32).reshape(-1, 1)
        except Exception:
            return None

    @staticmethod
    def from_dict(d):
        return ExtraTerm(d["term_type"], d["params"])


@dataclass
class Program:
    program_id: str = "?"
    n_lags: int = 5
    extra_terms: List[ExtraTerm] = field(default_factory=list)
    score: float = float("-inf")

    @staticmethod
    def from_dict(d):
        return Program(
            program_id  = d["program_id"],
            n_lags      = d["n_lags"],
            extra_terms = [ExtraTerm.from_dict(t) for t in d["extra_terms"]],
            score       = d["score"])


# ════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════
print("Loading data...")

ACTS = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "elu": nn.ELU}

class WeeklyAE(nn.Module):
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
    def encode(self, x): return self.encoder(x)

raw = pd.read_csv(DATA_CSV).sort_values(["province","year","week"]).reset_index(drop=True)
ID_COLS      = ["province", "year", "month", "week"]
FEATURE_COLS = [c for c in raw.columns if c not in ID_COLS]
raw["time_sin"] = np.sin(2.0 * np.pi * raw["week"] / 52.0)
raw["time_cos"] = np.cos(2.0 * np.pi * raw["week"] / 52.0)
INPUT_FEATURES = FEATURE_COLS + ["time_sin", "time_cos"]
INPUT_DIM      = len(INPUT_FEATURES)
train_mask     = raw["year"].isin(TRAIN_YEARS)
col_means      = raw.loc[train_mask, INPUT_FEATURES].mean()
raw[INPUT_FEATURES] = raw[INPUT_FEATURES].fillna(col_means)

ae = WeeklyAE(INPUT_DIM, LATENT_DIM, [512, 256], 0.158, "leakyrelu", True)
ae.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
ae.eval()

z_cols     = [f"z{i+1}" for i in range(LATENT_DIM)]
extra_mask = ~raw["year"].isin(TRAIN_YEARS)
X_extra    = torch.tensor(raw.loc[extra_mask, INPUT_FEATURES].values.astype(np.float32))
with torch.no_grad():
    Z_extra = ae.encode(X_extra).numpy()
extra_lat             = pd.DataFrame(Z_extra, columns=z_cols)
extra_lat["province"] = raw.loc[extra_mask, "province"].values
extra_lat["year"]     = raw.loc[extra_mask, "year"].values
extra_lat["week"]     = raw.loc[extra_mask, "week"].values

saved  = pd.read_csv(LATENT_CSV)
latent = pd.concat([saved, extra_lat], ignore_index=True)
latent = latent.merge(
    raw[["province","year","week","incidence"]],
    on=["province","year","week"], how="left")
latent = latent.sort_values(["province","year","week"]).reset_index(drop=True)

if LOG_INCIDENCE:
    latent["incidence"] = np.log1p(latent["incidence"].clip(lower=0))

clusters    = pd.read_csv(CLUSTER_CSV)[["province","cluster_id"]]
latent      = latent.merge(clusters, on="province", how="left")
CLUSTER_IDS = sorted(latent["cluster_id"].dropna().unique().astype(int))
STATE_COLS  = z_cols + ["incidence"]
STATE_DIM   = len(STATE_COLS)
INC_COL     = STATE_DIM - 1

with open(BEST_JSON) as f:
    program = Program.from_dict(json.load(f))

print(f"  Program: n_lags={program.n_lags}  extra_terms={len(program.extra_terms)}")

# Forecastable provinces (acf1 >= 0.75, computed on raw incidence)
raw_full = pd.read_csv(DATA_CSV).sort_values(["province","year","week"]).reset_index(drop=True)
fc_provs = set()
for prov, grp in raw_full.groupby("province"):
    inc = grp["incidence"].fillna(0).values
    var = float(np.var(inc))
    acf1 = float(np.corrcoef(inc[:-1], inc[1:])[0, 1]) if var > 1e-9 else 0.0
    if acf1 >= 0.75:
        fc_provs.add(prov)
print(f"  Forecastable provinces: {len(fc_provs)}/32\n")

# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════
poly = PolynomialFeatures(degree=POLY_DEG, include_bias=True)

def build_features(Z_sc, prog, weeks=None):
    T      = Z_sc.shape[0]
    n_lags = prog.n_lags
    if T <= n_lags + 1:
        return None, None
    lag_vecs = [Z_sc[n_lags - lag: T - 1 - lag] for lag in range(n_lags + 1)]
    X_lag    = np.concatenate(lag_vecs, axis=1).astype(np.float32)
    weeks_lag = (weeks[n_lags: T - 1].astype(np.float32)
                 if weeks is not None else None)
    extra_cols = []
    for term in prog.extra_terms:
        col = term.compute(X_lag, n_lags, STATE_DIM, INC_COL, weeks=weeks_lag)
        if col is not None and col.shape[0] == X_lag.shape[0]:
            extra_cols.append(col)
    if extra_cols:
        X_lag = np.concatenate([X_lag] + extra_cols, axis=1)
    Theta  = poly.fit_transform(X_lag).astype(np.float32)
    target = Z_sc[n_lags + 1:].astype(np.float32)
    return Theta, target

def build_dataset(df, year_list, scaler):
    all_theta, all_target = [], []
    for prov, grp in df[df["year"].isin(year_list)].groupby("province"):
        grp  = grp.sort_values(["year","week"]).reset_index(drop=True)
        Z_sc = scaler.transform(grp[STATE_COLS].values.astype(np.float32))
        Th, tgt = build_features(Z_sc, program, weeks=grp["week"].values)
        if Th is not None:
            all_theta.append(Th); all_target.append(tgt)
    if not all_theta:
        return None, None
    return np.vstack(all_theta), np.vstack(all_target)

def stlsq(Theta, Y, threshold, n_iters):
    W  = np.linalg.lstsq(Theta, Y, rcond=None)[0]
    tv = np.broadcast_to(np.atleast_1d(threshold), (Y.shape[1],))
    for _ in range(n_iters):
        for k in range(Y.shape[1]):
            small = np.abs(W[:, k]) < tv[k]
            W[small, k] = 0.0
            active = ~small
            if active.sum() == 0:
                continue
            W[active, k] = np.linalg.lstsq(Theta[:, active], Y[:, k], rcond=None)[0]
    return W

def province_r2(df, year_list, W_provinces, W_clusters, W_global, scaler):
    """Return list of per-province incidence R2 dicts."""
    results = []
    n_lags  = program.n_lags
    for prov, grp in df[df["year"].isin(year_list)].groupby("province"):
        grp_s  = grp.sort_values(["year","week"]).reset_index(drop=True)
        if len(grp_s) <= n_lags:
            continue
        Z_s    = grp_s[STATE_COLS].values.astype(np.float32)
        Z_s_sc = scaler.transform(Z_s)
        Th_s, Y_s = build_features(Z_s_sc, program, weeks=grp_s["week"].values)
        if Th_s is None:
            continue
        cid = int(latent.loc[latent["province"] == prov, "cluster_id"].iloc[0])
        W_p = W_provinces.get(prov, W_clusters.get(cid, W_global))
        Y_hat    = Th_s @ W_p
        Y_hat_r  = scaler.inverse_transform(Y_hat)
        Y_s_r    = scaler.inverse_transform(Y_s)
        if LOG_INCIDENCE:
            Y_hat_r[:, INC_COL] = np.expm1(Y_hat_r[:, INC_COL])
            Y_s_r[:,   INC_COL] = np.expm1(Y_s_r[:,   INC_COL])
        r2 = float(r2_score(Y_s_r[:, INC_COL], Y_hat_r[:, INC_COL]))
        results.append({"province": prov, "r2": r2})
    return results

# ════════════════════════════════════════════════════════════
# FIT GLOBAL EQUATION ONCE
# ════════════════════════════════════════════════════════════
train_rows = latent[latent["year"].isin(TRAIN_YEARS)][STATE_COLS].values
scaler     = StandardScaler()
scaler.fit(train_rows)

thresh_g = np.array([THRESH_GLOBAL] * LATENT_DIM + [THRESH_INC_GLOBAL])
thresh_p = np.array([THRESH_PROVINCE] * LATENT_DIM + [THRESH_INC_PROVINCE])

Theta_tr, Y_tr = build_dataset(latent, TRAIN_YEARS, scaler)
W_global = stlsq(Theta_tr, Y_tr, thresh_g, STLSQ_ITERS)
n_terms  = Theta_tr.shape[1]
print(f"Global equation fitted  ({n_terms} terms, "
      f"{int((W_global!=0).sum())} active)\n")

# ════════════════════════════════════════════════════════════
# THRESHOLD SWEEP
# ════════════════════════════════════════════════════════════
print(f"{'thresh':>8}  {'val_all32':>10}  {'val_fc22':>9}  "
      f"{'test_all32':>11}  {'test_fc22':>10}  "
      f"{'clusters_kept':>14}")
print("-" * 75)

sweep_rows = []

for tc in CLUSTER_THRESHOLDS:
    thresh_c = np.array([tc] * LATENT_DIM + [tc])   # same for latent and incidence

    # Fit cluster equations
    W_clusters: Dict[int, np.ndarray] = {}
    clusters_used = []
    for cid in CLUSTER_IDS:
        c_df = latent[latent["cluster_id"] == cid]
        Th_c, Y_c = build_dataset(c_df, TRAIN_YEARS, scaler)
        if Th_c is None:
            W_clusters[cid] = W_global.copy()
            continue
        Rc  = Y_c - Th_c @ W_global
        dWc = stlsq(Th_c, Rc, thresh_c, STLSQ_ITERS)
        W_clusters[cid] = W_global + dWc
        active = int((dWc != 0).sum())
        clusters_used.append(f"C{cid}:{active}")

    # Fit province equations
    W_provinces: Dict[str, np.ndarray] = {}
    for prov, grp in latent.groupby("province"):
        cid = int(grp["cluster_id"].iloc[0])
        W_c = W_clusters.get(cid, W_global)
        grp_tr = (grp[grp["year"].isin(TRAIN_YEARS)]
                  .sort_values(["year","week"]).reset_index(drop=True))
        Z_sc   = scaler.transform(grp_tr[STATE_COLS].values.astype(np.float32))
        Th_p, Y_p = build_features(Z_sc, program, weeks=grp_tr["week"].values)
        if Th_p is None or len(Th_p) < n_terms:
            W_provinces[prov] = W_c.copy()
        else:
            Rp  = Y_p - Th_p @ W_c
            dWp = stlsq(Th_p, Rp, thresh_p, STLSQ_ITERS)
            W_provinces[prov] = W_c + dWp

    # Evaluate
    val_rows  = province_r2(latent, VAL_YEARS,  W_provinces, W_clusters, W_global, scaler)
    test_rows = province_r2(latent, TEST_YEARS, W_provinces, W_clusters, W_global, scaler)

    val_all  = np.mean([r["r2"] for r in val_rows])
    test_all = np.mean([r["r2"] for r in test_rows])
    val_fc   = np.mean([r["r2"] for r in val_rows  if r["province"] in fc_provs])
    test_fc  = np.mean([r["r2"] for r in test_rows if r["province"] in fc_provs])

    marker = " <-- current" if abs(tc - 0.05) < 1e-9 else ""
    print(f"  {tc:>6.2f}  {val_all:>10.4f}  {val_fc:>9.4f}  "
          f"{test_all:>11.4f}  {test_fc:>10.4f}  "
          f"{' '.join(clusters_used)}{marker}")

    sweep_rows.append({
        "thresh_cluster": tc,
        "val_all32":  round(val_all,  4),
        "val_fc22":   round(val_fc,   4),
        "test_all32": round(test_all, 4),
        "test_fc22":  round(test_fc,  4),
    })

# Save sweep results
out = pd.DataFrame(sweep_rows)
out.to_csv(os.path.join(HERE, "sweep_cluster_threshold.csv"), index=False)

best_idx = out["val_fc22"].idxmax()
best_row = out.loc[best_idx]
print(f"\n  Best on val_fc22: thresh={best_row['thresh_cluster']:.2f}"
      f"  val_fc={best_row['val_fc22']:+.4f}"
      f"  test_fc={best_row['test_fc22']:+.4f}")
print(f"\n  Saved sweep_cluster_threshold.csv")
print("  DONE.")
