"""
run_epihsd_funsearch.py
────────────────────────────────────────────────────────────
Final evaluation of the best program discovered by FunSearch.

Loads funsearch_best.json, uses its n_lags and extra_terms to
build the EpiHSD feature matrix, trains on full TRAIN_YEARS
(2015-2019), then evaluates on the sealed VAL (2020-2021) and
TEST (2022-2023) splits.

Prints a side-by-side comparison with the baseline EpiHSD results
from epihsd_results.csv.

OUTPUTS (weekly-pipeline/)
──────────────────────────
  epihsd_funsearch_byregion.csv    per-province val and test R2
  epihsd_funsearch_results.csv     summary table all three levels
  epihsd_funsearch_equations.txt   discovered equations
────────────────────────────────────────────────────────────
"""

import os, json, copy
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
BASELINE_CSV = os.path.join(HERE, "epihsd_results.csv")

# ── EpiHSD settings (must match run_funsearch.py) ─────────
POLY_DEG            = 1
THRESH_GLOBAL       = 0.05
THRESH_CLUSTER      = 0.10
THRESH_PROVINCE     = 0.08
THRESH_INC_GLOBAL   = 0.04
THRESH_INC_CLUSTER  = 0.10
THRESH_INC_PROVINCE = 100.0
LOG_INCIDENCE       = True
STLSQ_ITERS         = 30

TRAIN_YEARS = list(range(2015, 2020))
VAL_YEARS   = [2020, 2021]
TEST_YEARS  = [2022, 2023]

# ════════════════════════════════════════════════════════════
# PROGRAM DATACLASSES (mirrors run_funsearch.py)
# ════════════════════════════════════════════════════════════

@dataclass
class ExtraTerm:
    term_type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def compute(self, X_lag: np.ndarray, n_lags: int,
                state_dim: int, inc_col: int,
                weeks: "np.ndarray | None" = None) -> "np.ndarray | None":
        try:
            tt = self.term_type
            p  = self.params
            ml = n_lags
            sd = state_dim
            if tt == "inc_pow":
                k  = min(p.get("lag", 0), ml)
                pw = p.get("power", 2)
                v  = X_lag[:, sd * k + inc_col] ** pw
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
                k  = min(p.get("lag", 0), ml)
                i  = min(p.get("z_idx1", 0), LATENT_DIM - 1)
                j  = min(p.get("z_idx2", 1), LATENT_DIM - 1)
                v  = X_lag[:, sd * k + i] * X_lag[:, sd * k + j]
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
                use_cos = p.get("use_cos", 0)
                angle   = 2.0 * np.pi * weeks / 52.0
                v = np.cos(angle) if use_cos else np.sin(angle)
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
    generation: int = 0
    parent_id: str = "?"

    @staticmethod
    def from_dict(d: dict) -> "Program":
        return Program(
            program_id  = d["program_id"],
            n_lags      = d["n_lags"],
            extra_terms = [ExtraTerm.from_dict(t) for t in d["extra_terms"]],
            score       = d["score"],
            generation  = d.get("generation", 0),
            parent_id   = d.get("parent_id", "?"))

    def describe(self) -> str:
        extras = [f"{t.term_type}({t.params})" for t in self.extra_terms]
        return (f"n_lags={self.n_lags}  "
                f"extras=[{', '.join(extras) if extras else 'none'}]  "
                f"inner_val_score={self.score:+.4f}")


# ════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════
print("Loading data and encoding latents...")

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

raw = pd.read_csv(DATA_CSV)
raw = raw.sort_values(["province", "year", "week"]).reset_index(drop=True)
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
    raw[["province", "year", "week", "incidence"]],
    on=["province", "year", "week"], how="left")
latent = latent.sort_values(["province", "year", "week"]).reset_index(drop=True)

if LOG_INCIDENCE:
    latent["incidence"] = np.log1p(latent["incidence"].clip(lower=0))

clusters   = pd.read_csv(CLUSTER_CSV)[["province", "cluster_id"]]
latent     = latent.merge(clusters, on="province", how="left")
CLUSTER_IDS = sorted(latent["cluster_id"].dropna().unique().astype(int))

STATE_COLS = z_cols + ["incidence"]
STATE_DIM  = len(STATE_COLS)
INC_COL    = STATE_DIM - 1

print(f"  Loaded {len(latent)} rows  "
      f"provinces={latent['province'].nunique()}  "
      f"clusters={len(CLUSTER_IDS)}\n")

# ════════════════════════════════════════════════════════════
# LOAD BEST PROGRAM
# ════════════════════════════════════════════════════════════
with open(BEST_JSON) as f:
    best_dict = json.load(f)
program = Program.from_dict(best_dict)

print("=" * 60)
print("  FUNSEARCH BEST PROGRAM")
print("=" * 60)
print(f"  {program.describe()}")
print(f"  Inner-val R2 (2018-2019): {program.score:+.4f}")
print()

# ════════════════════════════════════════════════════════════
# FEATURE BUILDER
# ════════════════════════════════════════════════════════════

poly = PolynomialFeatures(degree=POLY_DEG, include_bias=True)

def build_features(Z_sc: np.ndarray, prog: Program,
                   weeks: "np.ndarray | None" = None):
    """Build (Theta, target) for one province sequence using program spec.
    weeks: optional 1-D array of week-of-year values for every row of Z_sc.
    """
    T      = Z_sc.shape[0]
    n_lags = prog.n_lags
    if T <= n_lags + 1:
        return None, None
    lag_vecs = []
    for lag in range(n_lags + 1):
        lag_vecs.append(Z_sc[n_lags - lag: T - 1 - lag])
    X_lag = np.concatenate(lag_vecs, axis=1).astype(np.float32)
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


def build_dataset(df, year_list, scaler, prog):
    all_theta, all_target = [], []
    sel = df[df["year"].isin(year_list)]
    for prov, grp in sel.groupby("province"):
        grp  = grp.sort_values(["year", "week"]).reset_index(drop=True)
        Z_sc = scaler.transform(grp[STATE_COLS].values.astype(np.float32))
        Th, tgt = build_features(Z_sc, prog, weeks=grp["week"].values)
        if Th is None:
            continue
        all_theta.append(Th)
        all_target.append(tgt)
    if not all_theta:
        return None, None
    return np.vstack(all_theta), np.vstack(all_target)


# ════════════════════════════════════════════════════════════
# STLSQ
# ════════════════════════════════════════════════════════════

def stlsq(Theta, Y, threshold, n_iters):
    W = np.linalg.lstsq(Theta, Y, rcond=None)[0]
    tv = np.broadcast_to(np.atleast_1d(threshold), (Y.shape[1],))
    for _ in range(n_iters):
        for k in range(Y.shape[1]):
            small    = np.abs(W[:, k]) < tv[k]
            W[small, k] = 0.0
            active   = ~small
            if active.sum() == 0:
                continue
            W[active, k] = np.linalg.lstsq(
                Theta[:, active], Y[:, k], rcond=None)[0]
    return W


# ════════════════════════════════════════════════════════════
# FIT SCALER AND FEATURE SIZE
# ════════════════════════════════════════════════════════════
train_rows = latent[latent["year"].isin(TRAIN_YEARS)][STATE_COLS].values
scaler     = StandardScaler()
scaler.fit(train_rows)

thresh_g = np.array([THRESH_GLOBAL]   * LATENT_DIM + [THRESH_INC_GLOBAL])
thresh_c = np.array([THRESH_CLUSTER]  * LATENT_DIM + [THRESH_INC_CLUSTER])
thresh_p = np.array([THRESH_PROVINCE] * LATENT_DIM + [THRESH_INC_PROVINCE])

# Determine n_terms by probing one province
_probe_grp = (latent[latent["year"].isin(TRAIN_YEARS)]
              .groupby("province").first().reset_index())
_prov0 = latent[(latent["province"] == _probe_grp["province"].iloc[0]) &
                (latent["year"].isin(TRAIN_YEARS))].sort_values(["year","week"])
_Z0 = scaler.transform(_prov0[STATE_COLS].values.astype(np.float32))
_Th0, _ = build_features(_Z0, program, weeks=_prov0["week"].values)
n_terms = _Th0.shape[1]

print(f"  State dim       : {STATE_DIM}  ({STATE_COLS})")
print(f"  n_lags          : {program.n_lags}")
print(f"  extra terms     : {len(program.extra_terms)}")
print(f"  Feature columns : {n_terms}\n")

# ════════════════════════════════════════════════════════════
# LEVEL 1: GLOBAL
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("  LEVEL 1: Global equation  (all 32 provinces)")
print("=" * 60)

Theta_tr, Y_tr = build_dataset(latent, TRAIN_YEARS, scaler, program)
print(f"  Train pairs: {len(Theta_tr)}  Terms: {n_terms}"
      f"  Ratio: {len(Theta_tr)/n_terms:.1f}")

W_global  = stlsq(Theta_tr, Y_tr, thresh_g, STLSQ_ITERS)
active_g  = int((W_global != 0).sum())
print(f"  Active terms: {active_g} / {n_terms * STATE_DIM}")


def eval_split(df, year_list, W_provinces, W_clusters, W_global, scaler, prog):
    """Return list of {province, r2_latent, r2_incidence} dicts."""
    results = []
    n_lags  = prog.n_lags
    for prov, grp in df[df["year"].isin(year_list)].groupby("province"):
        grp_s  = grp.sort_values(["year", "week"]).reset_index(drop=True)
        if len(grp_s) <= n_lags:
            continue
        Z_s    = grp_s[STATE_COLS].values.astype(np.float32)
        Z_s_sc = scaler.transform(Z_s)
        Th_s, Y_s = build_features(Z_s_sc, prog, weeks=grp_s["week"].values)
        if Th_s is None:
            continue
        cid = int(latent.loc[latent["province"] == prov, "cluster_id"].iloc[0])
        W_p = W_provinces.get(prov, W_clusters.get(cid, W_global))
        Y_hat_sc  = Th_s @ W_p
        Y_hat_raw = scaler.inverse_transform(Y_hat_sc)
        Y_s_raw   = scaler.inverse_transform(Y_s)
        if LOG_INCIDENCE:
            Y_hat_raw[:, INC_COL] = np.expm1(Y_hat_raw[:, INC_COL])
            Y_s_raw[:,   INC_COL] = np.expm1(Y_s_raw[:,   INC_COL])
        r2_lat = float(r2_score(
            Y_s[:, :LATENT_DIM].flatten(),
            Y_hat_sc[:, :LATENT_DIM].flatten()))
        r2_inc = float(r2_score(
            Y_s_raw[:, INC_COL],
            Y_hat_raw[:, INC_COL]))
        results.append({"province": prov, "r2_latent": r2_lat,
                        "r2_incidence": r2_inc})
    return results


W_empty = {}
val_g  = eval_split(latent, VAL_YEARS,  W_empty, W_empty, W_global, scaler, program)
test_g = eval_split(latent, TEST_YEARS, W_empty, W_empty, W_global, scaler, program)
r2_vg_lat = np.mean([r["r2_latent"]    for r in val_g])
r2_tg_lat = np.mean([r["r2_latent"]    for r in test_g])
r2_vg_inc = np.mean([r["r2_incidence"] for r in val_g])
r2_tg_inc = np.mean([r["r2_incidence"] for r in test_g])
print(f"  Val  latent={r2_vg_lat:+.4f}  incidence={r2_vg_inc:+.4f}")
print(f"  Test latent={r2_tg_lat:+.4f}  incidence={r2_tg_inc:+.4f}\n")

# ════════════════════════════════════════════════════════════
# LEVEL 2: CLUSTER RESIDUALS
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("  LEVEL 2: Cluster residual equations  (val-selected)")
print("=" * 60)
print("  For each cluster the residual delta is only applied when it")
print("  improves val incidence R2 over the global equation. Clusters")
print("  where the delta hurts revert to the global equation.\n")

W_clusters: Dict[int, np.ndarray] = {}
results_lvl2 = []

for cid in CLUSTER_IDS:
    c_df   = latent[latent["cluster_id"] == cid]
    n_prov = c_df["province"].nunique()
    Th_c, Y_c = build_dataset(c_df, TRAIN_YEARS, scaler, program)
    if Th_c is None:
        W_clusters[cid] = W_global.copy()
        continue

    Rc  = Y_c - Th_c @ W_global
    dWc = stlsq(Th_c, Rc, thresh_c, STLSQ_ITERS)
    W_c = W_global + dWc
    active_c = int((dWc != 0).sum())

    # Val-based selection: keep cluster delta only if it beats global on val.
    # This uses the 2020-2021 val set which was never touched by FunSearch,
    # so there is no data leakage into the sealed 2022-2023 test set.
    val_g = eval_split(c_df, VAL_YEARS, {}, {cid: W_global}, W_global, scaler, program)
    val_c = eval_split(c_df, VAL_YEARS, {}, {cid: W_c},      W_global, scaler, program)
    r2_val_g = np.mean([r["r2_incidence"] for r in val_g]) if val_g else float("-inf")
    r2_val_c = np.mean([r["r2_incidence"] for r in val_c]) if val_c else float("-inf")

    if r2_val_c > r2_val_g:
        W_clusters[cid] = W_c
        selection = f"cluster  (val {r2_val_c:+.4f} > global {r2_val_g:+.4f})"
    else:
        W_clusters[cid] = W_global.copy()
        selection = f"REVERTED to global  (val {r2_val_c:+.4f} <= global {r2_val_g:+.4f})"

    # Evaluate final chosen equation on val and test
    W_chosen = W_clusters[cid]
    val_f  = eval_split(c_df, VAL_YEARS,  {}, {cid: W_chosen}, W_global, scaler, program)
    test_f = eval_split(c_df, TEST_YEARS, {}, {cid: W_chosen}, W_global, scaler, program)
    r2_vc_lat = np.mean([r["r2_latent"]    for r in val_f])  if val_f  else float("nan")
    r2_tc_lat = np.mean([r["r2_latent"]    for r in test_f]) if test_f else float("nan")
    r2_vc_inc = np.mean([r["r2_incidence"] for r in val_f])  if val_f  else float("nan")
    r2_tc_inc = np.mean([r["r2_incidence"] for r in test_f]) if test_f else float("nan")

    print(f"  Cluster {cid}  ({n_prov} prov)  delta_terms={active_c}  => {selection}")
    print(f"    latent    val={r2_vc_lat:+.4f}  test={r2_tc_lat:+.4f}")
    print(f"    incidence val={r2_vc_inc:+.4f}  test={r2_tc_inc:+.4f}")

    results_lvl2.append({
        "level": "cluster", "id": cid, "n_units": n_prov,
        "delta_terms": active_c,
        "selection":   "cluster" if r2_val_c > r2_val_g else "global",
        "val_r2_latent":  round(r2_vc_lat, 4),
        "test_r2_latent": round(r2_tc_lat, 4),
        "val_r2_inc":     round(r2_vc_inc, 4),
        "test_r2_inc":    round(r2_tc_inc, 4),
    })

# ════════════════════════════════════════════════════════════
# LEVEL 3: PROVINCE RESIDUALS
# ════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("  LEVEL 3: Province residual equations")
print("=" * 60)

W_provinces: Dict[str, np.ndarray] = {}
byregion = []

for prov, grp in latent.groupby("province"):
    cid = int(grp["cluster_id"].iloc[0])
    W_c = W_clusters.get(cid, W_global)
    grp_tr = (grp[grp["year"].isin(TRAIN_YEARS)]
              .sort_values(["year", "week"]).reset_index(drop=True))
    Z_sc   = scaler.transform(grp_tr[STATE_COLS].values.astype(np.float32))
    Th_p, Y_p = build_features(Z_sc, program, weeks=grp_tr["week"].values)
    if Th_p is None or len(Th_p) < n_terms:
        W_provinces[prov] = W_c.copy()
    else:
        Rp  = Y_p - Th_p @ W_c
        dWp = stlsq(Th_p, Rp, thresh_p, STLSQ_ITERS)
        W_provinces[prov] = W_c + dWp

    for split_name, year_list in [("val", VAL_YEARS), ("test", TEST_YEARS)]:
        grp_s  = (grp[grp["year"].isin(year_list)]
                  .sort_values(["year", "week"]).reset_index(drop=True))
        if len(grp_s) <= program.n_lags:
            continue
        Z_s    = grp_s[STATE_COLS].values.astype(np.float32)
        Z_s_sc = scaler.transform(Z_s)
        Th_s, Y_s = build_features(Z_s_sc, program, weeks=grp_s["week"].values)
        if Th_s is None:
            continue
        W_p      = W_provinces[prov]
        Y_hat    = Th_s @ W_p
        Y_hat_r  = scaler.inverse_transform(Y_hat)
        Y_s_r    = scaler.inverse_transform(Y_s)
        if LOG_INCIDENCE:
            Y_hat_r[:, INC_COL] = np.expm1(Y_hat_r[:, INC_COL])
            Y_s_r[:,   INC_COL] = np.expm1(Y_s_r[:,   INC_COL])
        r2_lat = float(r2_score(
            Y_s[:, :LATENT_DIM].flatten(),
            Y_hat[:, :LATENT_DIM].flatten()))
        r2_inc = float(r2_score(Y_s_r[:, INC_COL], Y_hat_r[:, INC_COL]))
        byregion.append({
            "province": prov, "cluster_id": cid,
            "split": split_name,
            "r2_latent": round(r2_lat, 4),
            "r2_incidence": round(r2_inc, 4),
        })

br_df    = pd.DataFrame(byregion)
val_lat  = br_df[br_df["split"] == "val"]["r2_latent"].mean()
test_lat = br_df[br_df["split"] == "test"]["r2_latent"].mean()
val_inc  = br_df[br_df["split"] == "val"]["r2_incidence"].mean()
test_inc = br_df[br_df["split"] == "test"]["r2_incidence"].mean()
print(f"  Province-level  latent    val={val_lat:+.4f}  test={test_lat:+.4f}")
print(f"  Province-level  incidence val={val_inc:+.4f}  test={test_inc:+.4f}")

# ════════════════════════════════════════════════════════════
# FORECASTABILITY CLASSIFICATION
# ════════════════════════════════════════════════════════════
# Criterion: lag-1 autocorrelation of the full incidence series >= 0.75.
# This is defined purely on data structure, independent of model results.
# Provinces below the threshold have near-memoryless dynamics that no
# lag-based model can reliably forecast, regardless of feature library.
ACF_THRESHOLD = 0.75

raw_full = pd.read_csv(DATA_CSV).sort_values(
    ["province", "year", "week"]).reset_index(drop=True)

fc_records = []
for prov, grp in raw_full.groupby("province"):
    inc  = grp["incidence"].fillna(0).values
    var  = float(np.var(inc))
    acf1 = float(np.corrcoef(inc[:-1], inc[1:])[0, 1]) if var > 1e-9 else 0.0
    zero_pct = float((inc < 0.01).mean() * 100)
    forecastable = acf1 >= ACF_THRESHOLD
    fc_records.append({
        "province":     prov,
        "acf1":         round(acf1, 4),
        "zero_pct":     round(zero_pct, 1),
        "forecastable": forecastable,
        "reason":       "ok" if forecastable else "acf1_below_threshold",
    })

fc_df = pd.DataFrame(fc_records)
forecastable_provs = set(fc_df[fc_df["forecastable"]]["province"])
excluded_provs     = set(fc_df[~fc_df["forecastable"]]["province"])

print("\n" + "=" * 60)
print(f"  FORECASTABILITY FILTER  (acf1 >= {ACF_THRESHOLD})")
print("=" * 60)
print(f"  Forecastable provinces : {len(forecastable_provs)} / {len(fc_df)}")
print(f"  Excluded  provinces    : {len(excluded_provs)}")
print(f"\n  Excluded (low temporal persistence):")
excl_rows = fc_df[~fc_df["forecastable"]].sort_values("acf1")
for _, r in excl_rows.iterrows():
    print(f"    {r['province']:<35s}  acf1={r['acf1']:.3f}  zeros={r['zero_pct']:.1f}%")

# Filtered province-level metrics (test only, forecastable subset)
test_all  = br_df[br_df["split"] == "test"].copy()
test_fc   = test_all[test_all["province"].isin(forecastable_provs)]
val_all   = br_df[br_df["split"] == "val"].copy()
val_fc    = val_all[val_all["province"].isin(forecastable_provs)]

val_inc_fc  = val_fc["r2_incidence"].mean()
test_inc_fc = test_fc["r2_incidence"].mean()
val_lat_fc  = val_fc["r2_latent"].mean()
test_lat_fc = test_fc["r2_latent"].mean()

print(f"\n  Forecastable subset ({len(forecastable_provs)} prov):")
print(f"    incidence  val={val_inc_fc:+.4f}  test={test_inc_fc:+.4f}")
print(f"    latent     val={val_lat_fc:+.4f}  test={test_lat_fc:+.4f}")

# ════════════════════════════════════════════════════════════
# COMPARISON TABLE vs BASELINE EpiHSD
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  FUNSEARCH vs BASELINE EpiHSD")
print("=" * 60)

try:
    base   = pd.read_csv(BASELINE_CSV)
    base_g = base[base["level"] == "global"].iloc[0]
    base_p = base[base["level"] == "province"].iloc[0]

    # Compute baseline filtered metrics from epihsd_byregion.csv
    bl_br = pd.read_csv(os.path.join(HERE, "epihsd_byregion.csv"))
    bl_test_fc = bl_br[(bl_br["split"] == "test") &
                       (bl_br["province"].isin(forecastable_provs))]
    bl_val_fc  = bl_br[(bl_br["split"] == "val") &
                       (bl_br["province"].isin(forecastable_provs))]
    bl_inc_fc_test = bl_test_fc["r2_incidence"].mean()
    bl_inc_fc_val  = bl_val_fc["r2_incidence"].mean()
    bl_lat_fc_test = bl_test_fc["r2_latent"].mean()
    bl_lat_fc_val  = bl_val_fc["r2_latent"].mean()

    hdr = f"  {'':34s}  {'INCIDENCE R2':>22s}  {'LATENT R2':>22s}"
    sep = f"  {'':34s}  {'val':>10s}  {'test':>10s}  {'val':>10s}  {'test':>10s}"
    div = f"  {'-'*34}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}"

    print(f"\n  ALL 32 PROVINCES")
    print(hdr); print(sep); print(div)
    print(f"  {'Global   baseline':34s}"
          f"  {base_g['val_r2_inc']:+.4f}    {base_g['test_r2_inc']:+.4f}"
          f"    {base_g['val_r2_latent']:+.4f}    {base_g['test_r2_latent']:+.4f}")
    print(f"  {'Global   FunSearch':34s}"
          f"  {r2_vg_inc:+.4f}    {r2_tg_inc:+.4f}"
          f"    {r2_vg_lat:+.4f}    {r2_tg_lat:+.4f}")
    print(f"  {'  delta (test)':34s}  {'':10s}  "
          f"{r2_tg_inc - base_g['test_r2_inc']:+.4f}  {'':10s}")
    print()
    print(f"  {'Province baseline (all 32)':34s}"
          f"  {base_p['val_r2_inc']:+.4f}    {base_p['test_r2_inc']:+.4f}"
          f"    {base_p['val_r2_latent']:+.4f}    {base_p['test_r2_latent']:+.4f}")
    print(f"  {'Province FunSearch (all 32)':34s}"
          f"  {val_inc:+.4f}    {test_inc:+.4f}"
          f"    {val_lat:+.4f}    {test_lat:+.4f}")
    print(f"  {'  delta (test)':34s}  {'':10s}  "
          f"{test_inc - base_p['test_r2_inc']:+.4f}  {'':10s}")

    print(f"\n  FORECASTABLE SUBSET  ({len(forecastable_provs)} provinces, acf1 >= {ACF_THRESHOLD})")
    print(hdr); print(sep); print(div)
    print(f"  {'Province baseline (forecastable)':34s}"
          f"  {bl_inc_fc_val:+.4f}    {bl_inc_fc_test:+.4f}"
          f"    {bl_lat_fc_val:+.4f}    {bl_lat_fc_test:+.4f}")
    print(f"  {'Province FunSearch (forecastable)':34s}"
          f"  {val_inc_fc:+.4f}    {test_inc_fc:+.4f}"
          f"    {val_lat_fc:+.4f}    {test_lat_fc:+.4f}")
    print(f"  {'  delta (test)':34s}  {'':10s}  "
          f"{test_inc_fc - bl_inc_fc_test:+.4f}  {'':10s}")

    print(f"\n  Inner-val score used by FunSearch : {program.score:+.4f}")
    print(f"  (Inner-val = 2018-2019 incidence R2, fit on 2015-2017)")

except Exception as e:
    print(f"  Could not load baseline: {e}")

# Per-province test breakdown with forecastability flag
print("\n  Per-province test R2 (Level 3, FunSearch):")
test_prov = test_all.merge(fc_df[["province","acf1","forecastable"]], on="province")
for cid in sorted(test_prov["cluster_id"].unique()):
    rows = (test_prov[test_prov["cluster_id"] == cid]
            .sort_values("r2_incidence", ascending=False))
    print(f"\n  Cluster {int(cid)}:")
    print(f"    {'Province':<35s}  {'acf1':>6}  {'fc':>4}  "
          f"{'latent':>8}  {'incidence':>9}")
    for _, row in rows.iterrows():
        fc_flag = "yes" if row["forecastable"] else "NO"
        print(f"    {row['province']:<35s}  {row['acf1']:>6.3f}  {fc_flag:>4}  "
              f"{row['r2_latent']:+.4f}    {row['r2_incidence']:+.4f}")

# ════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ════════════════════════════════════════════════════════════
br_df.to_csv(os.path.join(HERE, "epihsd_funsearch_byregion.csv"), index=False)
fc_df.to_csv(os.path.join(HERE, "epihsd_funsearch_forecastability.csv"), index=False)

summary_rows = [
    {"level": "global", "id": "all", "n_units": 32,
     "val_r2_latent":  round(r2_vg_lat, 4),
     "test_r2_latent": round(r2_tg_lat, 4),
     "val_r2_inc":     round(r2_vg_inc, 4),
     "test_r2_inc":    round(r2_tg_inc, 4)}
] + results_lvl2 + [
    {"level": "province_all32", "id": "all", "n_units": 32,
     "val_r2_latent":  round(val_lat,     4),
     "test_r2_latent": round(test_lat,    4),
     "val_r2_inc":     round(val_inc,     4),
     "test_r2_inc":    round(test_inc,    4)},
    {"level": f"province_forecastable{len(forecastable_provs)}",
     "id": f"acf1_gte_{ACF_THRESHOLD}",
     "n_units": len(forecastable_provs),
     "val_r2_latent":  round(val_lat_fc,  4),
     "test_r2_latent": round(test_lat_fc, 4),
     "val_r2_inc":     round(val_inc_fc,  4),
     "test_r2_inc":    round(test_inc_fc, 4)},
]
pd.DataFrame(summary_rows).to_csv(
    os.path.join(HERE, "epihsd_funsearch_results.csv"), index=False)

# Save equations
feat_names = ([f"lag{lag}_{col}"
               for lag in range(program.n_lags + 1)
               for col in STATE_COLS]
              + [f"extra_{i}_{t.term_type}"
                 for i, t in enumerate(program.extra_terms)])
try:
    poly_names = poly.get_feature_names_out(feat_names)
except Exception:
    poly_names = [f"f{i}" for i in range(n_terms)]

eq_lines = [
    f"EpiHSD FunSearch EQUATIONS\n",
    f"  state_dim={STATE_DIM}  n_lags={program.n_lags}  "
    f"extra_terms={len(program.extra_terms)}\n",
    f"  inner_val_score={program.score:+.4f}\n",
    f"  forecastable_provinces={len(forecastable_provs)}/{len(fc_df)}"
    f"  (acf1 >= {ACF_THRESHOLD})\n\n",
    "EXTRA TERMS DISCOVERED BY FUNSEARCH\n",
]
for t in program.extra_terms:
    eq_lines.append(f"  {t.term_type:15s}  {t.params}\n")
eq_lines.append(f"\nEXCLUDED PROVINCES (acf1 < {ACF_THRESHOLD})\n")
for _, r in excl_rows.iterrows():
    eq_lines.append(f"  {r['province']}  acf1={r['acf1']}\n")
eq_lines.append("\nLEVEL 1: GLOBAL EQUATION\n")
for d, oname in enumerate(STATE_COLS):
    terms = [(poly_names[j], W_global[j, d])
             for j in range(n_terms) if W_global[j, d] != 0]
    eq = " + ".join(f"{v:.5f}*{n}" for n, v in terms) if terms else "0"
    eq_lines.append(f"  {oname}(t+1) = {eq}\n")

with open(os.path.join(HERE, "epihsd_funsearch_equations.txt"), "w") as f:
    f.writelines(eq_lines)

print(f"\n  Saved epihsd_funsearch_byregion.csv")
print(f"  Saved epihsd_funsearch_results.csv")
print(f"  Saved epihsd_funsearch_forecastability.csv")
print(f"  Saved epihsd_funsearch_equations.txt")
print("\n  DONE.")
print("-" * 60)
