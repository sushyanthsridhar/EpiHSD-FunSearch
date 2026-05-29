"""
run_epihsd_weekly.py
────────────────────────────────────────────────────────────────
EpiHSD: Epidemic Hierarchical Sparse Dynamics

A novel three-level residual sparse autoregressive framework
for discovering interpretable epidemic equations in latent space.

ALGORITHM
─────────
Level 1 (Global):
  Fit a sparse polynomial autoregression on ALL provinces.
  Feature matrix Theta = poly(z_t, z_{t-1}, ..., z_{t-L}).
  Coefficients W_global via STLSQ (sparse least squares).
  This captures universal dengue dynamics across the DR.

Level 2 (Cluster):
  For each cluster c, compute residuals after applying W_global.
  Fit sparse delta_W_c to those residuals.
  Final cluster equation: W_c = W_global + delta_W_c.
  delta_W_c encodes regional effects not captured globally.
  These deviations ARE the ghost variables discussed in the paper.

Level 3 (Province):
  For each province p in cluster c, compute residuals after W_c.
  Fit sparse delta_W_p to those residuals.
  Final province equation: W_p = W_c + delta_W_p.
  delta_W_p captures idiosyncratic local transmission dynamics.

INSPIRATIONS
────────────
  SINDy (Brunton, Proctor, Kutz 2016)          sparse equation discovery
  Gradient Boosting (Freund, Schapire 1997)     residual fitting at each level
  Hierarchical Linear Models (Gelman, Hill 2006) epidemiological hierarchy
  VAR / ARIMAX                                   lagged autoregressive features

KEY NOVELTY OVER PLAIN SINDy
─────────────────────────────
  Plain SINDy used only z_t as features (ratio 1.8, catastrophic).
  EpiHSD uses z_t, z_{t-1}, ..., z_{t-L} as a joint feature matrix
  (ratio 40-148, healthy). The residual hierarchy reduces effective
  degrees of freedom at each level, preventing overfitting.

OUTPUTS (weekly-pipeline/)
──────────────────────────
  epihsd_equations.txt       discovered equations at all three levels
  epihsd_results.csv         R2 at each level (global, cluster, province)
  epihsd_byregion.csv        per-province val and test R2
  epihsd_coefficients.csv    coefficient matrix for analysis and paper
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

LATENT_DIM  = 6
LATENT_CSV  = os.path.join(HERE, "latents", f"latent_weekly_dim{LATENT_DIM}.csv")
MODEL_PATH  = os.path.join(HERE, "models", f"best_model_weekly_dim{LATENT_DIM}.pt")
CLUSTER_CSV = os.path.join(HERE, "province_clusters.csv")

# ── EpiHSD CONFIG ─────────────────────────────────────────────
N_LAGS         = 5      # number of lag timesteps for state vector
N_RAIN_LAGS    = 3      # avg_rainfall lags to add as exogenous inputs (lag1..lag3)
POLY_DEG       = 1      # polynomial degree on lag features
THRESH_GLOBAL  = 0.05   # STLSQ threshold for global equation, latent dims
THRESH_CLUSTER = 0.05   # STLSQ threshold for cluster residuals, latent dims
THRESH_PROVINCE= 0.08   # slightly higher to prevent province overfitting, latent dims
THRESH_INC_GLOBAL   = 0.04  # global has 8192 rows and ratio 189, can afford more inc terms
THRESH_INC_CLUSTER  = 0.04  # cluster has 500+ rows, can afford more inc terms
THRESH_INC_PROVINCE = 100.0 # effectively no province delta for incidence: province delta
                            # only refines z1..z6, cluster equation is the final incidence word
USE_EXOG       = False  # off: seasonal and rainfall already encoded in z1..z6
LOG_INCIDENCE  = True   # log1p-transform incidence in state vector, expm1 when reporting R2
USE_SEIR_TERMS = False  # off: too many features for small clusters like Cluster 0
STLSQ_ITERS    = 30

TRAIN_YEARS = list(range(2015, 2020))
VAL_YEARS   = [2020, 2021]
TEST_YEARS  = [2022, 2023]

print("EpiHSD: Epidemic Hierarchical Sparse Dynamics")
print(f"  latent_dim={LATENT_DIM}  n_lags={N_LAGS}  poly_deg={POLY_DEG}")
print(f"  thresh global={THRESH_GLOBAL}  cluster={THRESH_CLUSTER}  province={THRESH_PROVINCE}\n")

# ── Encode test rows ──────────────────────────────────────────
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
    def forward(self, x): return self.decoder(self.encoder(x))
    def encode(self, x):  return self.encoder(x)
    def decode(self, z):  return self.decoder(z)

raw = pd.read_csv(DATA_CSV)
raw = raw.sort_values(["province", "year", "week"]).reset_index(drop=True)
ID_COLS      = ["province", "year", "month", "week"]
FEATURE_COLS = [c for c in raw.columns if c not in ID_COLS]
raw["time_sin"] = np.sin(2.0 * np.pi * raw["week"] / 52.0)
raw["time_cos"] = np.cos(2.0 * np.pi * raw["week"] / 52.0)
INPUT_FEATURES = FEATURE_COLS + ["time_sin", "time_cos"]
INPUT_DIM      = len(INPUT_FEATURES)
train_mask = raw["year"].isin(TRAIN_YEARS)
col_means  = raw.loc[train_mask, INPUT_FEATURES].mean()
raw[INPUT_FEATURES] = raw[INPUT_FEATURES].fillna(col_means)

ae = WeeklyAE(INPUT_DIM, LATENT_DIM, [512, 256], 0.158, "leakyrelu", True)
ae.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
ae.eval()
test_mask = raw["year"].isin(TEST_YEARS)
X_test_raw = torch.tensor(raw.loc[test_mask, INPUT_FEATURES].values.astype(np.float32))
with torch.no_grad():
    Z_test = ae.encode(X_test_raw).numpy()

z_cols     = [f"z{i+1}" for i in range(LATENT_DIM)]
# ── EpiHSD state vector: latent dims + incidence as a direct output ──
# This avoids the lossy autoencoder decode path.
# Incidence R2 is now read from the 7th output column of W directly.
STATE_COLS = z_cols + ["incidence"]
STATE_DIM  = len(STATE_COLS)   # 7
INC_COL    = STATE_DIM - 1     # index of incidence in the state vector
test_latent = pd.DataFrame(Z_test, columns=z_cols)
test_latent["province"] = raw.loc[test_mask, "province"].values
test_latent["year"]     = raw.loc[test_mask, "year"].values
test_latent["week"]     = raw.loc[test_mask, "week"].values
test_latent["split"]    = "test"

saved  = pd.read_csv(LATENT_CSV)
latent = pd.concat([saved, test_latent], ignore_index=True)
latent = latent.merge(
    raw[["province", "year", "week", "incidence", "avg_rainfall",
         "time_sin", "time_cos"]],
    on=["province", "year", "week"], how="left"
)
latent = latent.sort_values(["province", "year", "week"]).reset_index(drop=True)

# Log-transform incidence so epidemic dynamics are approximately linear in log space.
# Predicting log1p(incidence) avoids large outbreak spikes dominating the MSE.
# When reporting R2 we expm1 back to original scale for a fair comparison with baselines.
if LOG_INCIDENCE:
    latent["incidence"] = np.log1p(latent["incidence"].clip(lower=0))

clusters   = pd.read_csv(CLUSTER_CSV)[["province", "cluster_id"]]
latent     = latent.merge(clusters, on="province", how="left")

# ── EpiHSD core helpers ───────────────────────────────────────
poly = PolynomialFeatures(degree=POLY_DEG, include_bias=True)

def make_exog(grp):
    """
    Build exogenous feature matrix for one province group (already sorted).
    Rows align with the full T-length sequence.
    Columns: time_sin[t], time_cos[t], rain[t-1], rain[t-2], rain[t-3].
    At rows where a rainfall lag would go out of bounds we use the earliest
    available value (forward fill from the start).
    """
    T    = len(grp)
    tsin = grp["time_sin"].values.astype(np.float32)
    tcos = grp["time_cos"].values.astype(np.float32)
    rain = grp["avg_rainfall"].values.astype(np.float32)
    cols = [tsin, tcos]
    for lag in range(1, N_RAIN_LAGS + 1):
        pad    = np.full(lag, rain[0], dtype=np.float32)
        lagged = np.concatenate([pad, rain[:-lag]])
        cols.append(lagged)
    return np.column_stack(cols)   # (T, 2 + N_RAIN_LAGS)

N_EXOG = 2 + N_RAIN_LAGS   # time_sin, time_cos, rain_lag1..N_RAIN_LAGS

def build_lag_features(Z_seq, exog_seq=None):
    """
    Z_seq   : (T, STATE_DIM) state sequence, used for both lag inputs and targets.
    exog_seq: (T, N_EXOG) optional exogenous features appended at current time t.
              These are NOT lagged again here, just sliced at the current timestep.
    SEIR terms (when USE_SEIR_TERMS): inc_lag1 * z1_lag1 .. inc_lag1 * z6_lag1 and
    inc_lag1^2. These capture epidemic-state coupling and saturation with just 7 terms.
    Returns Theta (T-N_LAGS-1, n_terms) and target (T-N_LAGS-1, STATE_DIM).
    """
    T, K = Z_seq.shape
    if T <= N_LAGS + 1:
        return None, None
    lag_vecs = []
    for lag in range(N_LAGS + 1):
        lag_vecs.append(Z_seq[N_LAGS - lag: T - 1 - lag])
    X_lag = np.concatenate(lag_vecs, axis=1)       # (n, K*(N_LAGS+1))
    if exog_seq is not None:
        exog_t = exog_seq[N_LAGS: T - 1]            # current-t exog, (n, N_EXOG)
        X_lag  = np.concatenate([X_lag, exog_t], axis=1)
    if USE_SEIR_TERMS:
        # lag1 group starts at column STATE_DIM * 1
        inc_lag1 = X_lag[:, STATE_DIM * 1 + INC_COL : STATE_DIM * 1 + INC_COL + 1]  # (n,1)
        z_lag1   = X_lag[:, STATE_DIM * 1 : STATE_DIM * 1 + LATENT_DIM]             # (n,6)
        seir     = np.concatenate([inc_lag1 * z_lag1, inc_lag1 ** 2], axis=1)        # (n,7)
        X_lag    = np.concatenate([X_lag, seir], axis=1)
    Theta  = poly.fit_transform(X_lag)
    target = Z_seq[N_LAGS + 1:]
    return Theta.astype(np.float32), target.astype(np.float32)

def build_dataset(df, year_list, scaler=None):
    """
    Build (Theta, target) from all provinces in df for given years.
    Pairs are WITHIN each province only.
    Exogenous features (time, rainfall lags) are appended automatically.
    """
    all_theta, all_target = [], []
    sel = df[df["year"].isin(year_list)]
    for prov, grp in sel.groupby("province"):
        grp  = grp.sort_values(["year", "week"]).reset_index(drop=True)
        Z    = grp[STATE_COLS].values.astype(np.float32)
        if scaler is not None:
            Z = scaler.transform(Z)
        exog = make_exog(grp) if USE_EXOG else None
        Th, tgt = build_lag_features(Z, exog)
        if Th is None:
            continue
        all_theta.append(Th)
        all_target.append(tgt)
    if not all_theta:
        return None, None
    return np.vstack(all_theta), np.vstack(all_target)

def stlsq(Theta, Y, threshold, n_iters):
    """
    Sequentially-Thresholded Least Squares with per-output threshold support.
    Theta    : (n, p)
    Y        : (n, K)
    threshold: scalar applied to all outputs, or array of shape (K,) for
               per-output thresholds. Use a lower value for the incidence
               output so the equation is allowed more active terms.
    Returns W: (p, K)
    """
    W = np.linalg.lstsq(Theta, Y, rcond=None)[0]
    thresh_vec = np.broadcast_to(np.atleast_1d(threshold), (Y.shape[1],))
    for _ in range(n_iters):
        for k in range(Y.shape[1]):
            small_k  = np.abs(W[:, k]) < thresh_vec[k]
            W[small_k, k] = 0.0
            active_k = ~small_k
            if active_k.sum() == 0:
                continue
            W[active_k, k] = np.linalg.lstsq(
                Theta[:, active_k], Y[:, k], rcond=None)[0]
    return W

def predict_rollout(df, year_list, W, scaler):
    """
    Roll out the equation W one step at a time per province.
    State vector is STATE_COLS = [z1..z6, incidence] (7-dim).
    W has shape (n_terms, 7), so incidence is predicted directly
    as the 7th output — no decoder needed.

    r2_latent   : R2 on z1..z6 (columns 0:6), flattened.
    r2_incidence: R2 on incidence (column 6), inverse-transformed
                  back to original scale for fair comparison with
                  the ML-benchmark baselines.
    """
    results = []
    sel = df[df["year"].isin(year_list)]
    for prov, grp in sel.groupby("province"):
        grp  = grp.sort_values(["year", "week"]).reset_index(drop=True)
        S    = grp[STATE_COLS].values.astype(np.float32)   # (T, 7)
        S_sc = scaler.transform(S)
        exog = make_exog(grp) if USE_EXOG else None        # (T, N_EXOG) or None
        T    = len(S_sc)
        if T <= N_LAGS:
            continue

        preds = []
        for t in range(N_LAGS, T - 1):
            lag_vecs = [S_sc[t - lag] for lag in range(N_LAGS + 1)]
            if exog is not None:
                x_lag = np.concatenate(lag_vecs + [exog[t]]).reshape(1, -1)
            else:
                x_lag = np.concatenate(lag_vecs).reshape(1, -1)
            if USE_SEIR_TERMS:
                # lag1 vector is the second element in lag_vecs (lag=1)
                inc_lag1 = S_sc[t - 1, INC_COL]
                z_lag1   = S_sc[t - 1, :LATENT_DIM]
                seir     = np.concatenate([inc_lag1 * z_lag1,
                                           [inc_lag1 ** 2]]).reshape(1, -1)
                x_lag    = np.concatenate([x_lag, seir], axis=1)
            theta_t  = poly.transform(x_lag)
            s_hat    = (theta_t @ W).flatten()             # (7,)
            preds.append(s_hat)

        S_pred_sc = np.array(preds)                        # (n, 7) scaled
        S_true_sc = S_sc[N_LAGS + 1:]                      # (n, 7) scaled

        # Latent R2: on z1..z6 only (columns 0:6)
        r2_lat = float(r2_score(
            S_true_sc[:, :LATENT_DIM].flatten(),
            S_pred_sc[:, :LATENT_DIM].flatten()))

        # Incidence R2: inverse-transform column 6 back to original scale,
        # then expm1 if we stored log1p(incidence) in the state vector.
        S_pred_raw = scaler.inverse_transform(S_pred_sc)
        S_true_raw = scaler.inverse_transform(S_true_sc)
        if LOG_INCIDENCE:
            S_pred_raw[:, INC_COL] = np.expm1(S_pred_raw[:, INC_COL])
            S_true_raw[:, INC_COL] = np.expm1(S_true_raw[:, INC_COL])
        r2_inc = float(r2_score(
            S_true_raw[:, INC_COL],
            S_pred_raw[:, INC_COL]))

        results.append({
            "province"     : prov,
            "r2_latent"    : round(r2_lat, 4),
            "r2_incidence" : round(r2_inc, 4),
        })
    return results

# ── Fit scaler on all train data (7-dim state: z1..z6 + incidence) ───
train_rows = latent[latent["year"].isin(TRAIN_YEARS)][STATE_COLS].values
scaler     = StandardScaler()
scaler.fit(train_rows)
n_exog_active = N_EXOG if USE_EXOG else 0
n_seir_active = (LATENT_DIM + 1) if USE_SEIR_TERMS else 0
n_raw   = STATE_DIM * (N_LAGS + 1) + n_exog_active + n_seir_active
poly.fit(np.zeros((1, n_raw)))                       # initialise feature names

dummy_exog = np.zeros((N_LAGS + 2, N_EXOG)) if USE_EXOG else None
n_terms    = build_lag_features(
    np.zeros((N_LAGS + 2, STATE_DIM)), dummy_exog)[0].shape[1]
print(f"  State vector : {STATE_COLS}")
if USE_EXOG:
    print(f"  Exog features: time_sin, time_cos, avg_rainfall lag1..{N_RAIN_LAGS}")
else:
    print(f"  Exog features: none  (USE_EXOG=False)")
print(f"  Log incidence: {LOG_INCIDENCE}")
print(f"  SEIR terms   : {USE_SEIR_TERMS}  ({n_seir_active} extra interaction terms)")
print(f"  Feature matrix: {STATE_DIM} dims x {N_LAGS+1} lags + {n_exog_active} exog"
      f" + {n_seir_active} seir = {n_raw} raw features -> {n_terms} poly terms\n")

# Per-output STLSQ threshold vectors.
# Incidence output gets a lower threshold to allow more active terms.
thresh_global_vec   = np.array([THRESH_GLOBAL]   * LATENT_DIM + [THRESH_INC_GLOBAL])
thresh_cluster_vec  = np.array([THRESH_CLUSTER]  * LATENT_DIM + [THRESH_INC_CLUSTER])
thresh_province_vec = np.array([THRESH_PROVINCE] * LATENT_DIM + [THRESH_INC_PROVINCE])

# ════════════════════════════════════════════════════════════════
# LEVEL 1: GLOBAL EQUATION
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("  LEVEL 1: Global equation  (all 32 provinces)")
print("=" * 60)

Theta_tr, Y_tr = build_dataset(latent, TRAIN_YEARS, scaler)
print(f"  Train pairs: {len(Theta_tr)}  Terms: {n_terms}"
      f"  Ratio: {len(Theta_tr)/n_terms:.1f}")

W_global = stlsq(Theta_tr, Y_tr, thresh_global_vec, STLSQ_ITERS)
active_g = int((W_global != 0).sum())
print(f"  Active terms: {active_g} / {n_terms * STATE_DIM}")

val_rows_g  = predict_rollout(latent, VAL_YEARS,  W_global, scaler)
test_rows_g = predict_rollout(latent, TEST_YEARS, W_global, scaler)
r2_val_g      = float(np.mean([r["r2_latent"]    for r in val_rows_g]))
r2_test_g     = float(np.mean([r["r2_latent"]    for r in test_rows_g]))
r2_val_g_inc  = float(np.mean([r["r2_incidence"] for r in val_rows_g]))
r2_test_g_inc = float(np.mean([r["r2_incidence"] for r in test_rows_g]))
print(f"  Val  latent_r2={r2_val_g:+.4f}   incidence_r2={r2_val_g_inc:+.4f}")
print(f"  Test latent_r2={r2_test_g:+.4f}   incidence_r2={r2_test_g_inc:+.4f}\n")

# ════════════════════════════════════════════════════════════════
# LEVEL 2: CLUSTER RESIDUAL EQUATIONS
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("  LEVEL 2: Cluster residual equations  (5 clusters)")
print("=" * 60)

cluster_ids = sorted(latent["cluster_id"].dropna().unique())
W_clusters  = {}
results_lvl2 = []

for cid in cluster_ids:
    cid  = int(cid)
    c_df = latent[latent["cluster_id"] == cid]
    c_provs = sorted(c_df["province"].unique())

    Theta_c, Y_c = build_dataset(c_df, TRAIN_YEARS, scaler)
    if Theta_c is None:
        W_clusters[cid] = W_global.copy()
        continue

    # Residuals after applying global equation
    R_c = Y_c - Theta_c @ W_global
    delta_W_c = stlsq(Theta_c, R_c, thresh_cluster_vec, STLSQ_ITERS)
    W_c       = W_global + delta_W_c
    W_clusters[cid] = W_c

    active_c = int((delta_W_c != 0).sum())
    val_rows_c  = predict_rollout(c_df, VAL_YEARS,  W_c, scaler)
    test_rows_c = predict_rollout(c_df, TEST_YEARS, W_c, scaler)
    r2_val_c      = float(np.mean([r["r2_latent"]    for r in val_rows_c]))  if val_rows_c  else float("nan")
    r2_test_c     = float(np.mean([r["r2_latent"]    for r in test_rows_c])) if test_rows_c else float("nan")
    r2_val_c_inc  = float(np.mean([r["r2_incidence"] for r in val_rows_c]))  if val_rows_c  else float("nan")
    r2_test_c_inc = float(np.mean([r["r2_incidence"] for r in test_rows_c])) if test_rows_c else float("nan")

    print(f"  Cluster {cid}  ({len(c_provs)} prov)  delta_terms={active_c}")
    print(f"    latent    val={r2_val_c:+.4f}  test={r2_test_c:+.4f}")
    print(f"    incidence val={r2_val_c_inc:+.4f}  test={r2_test_c_inc:+.4f}")

    results_lvl2.append({
        "level"         : "cluster",
        "id"            : cid,
        "n_units"       : len(c_provs),
        "delta_terms"   : active_c,
        "val_r2_latent" : round(r2_val_c,     4),
        "test_r2_latent": round(r2_test_c,    4),
        "val_r2_inc"    : round(r2_val_c_inc, 4),
        "test_r2_inc"   : round(r2_test_c_inc,4),
    })

# ════════════════════════════════════════════════════════════════
# LEVEL 3: PROVINCE RESIDUAL EQUATIONS
# ════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("  LEVEL 3: Province residual equations")
print("=" * 60)

W_provinces  = {}
results_lvl3 = []
byregion     = []

for prov, grp in latent.groupby("province"):
    cid = int(grp["cluster_id"].iloc[0])
    W_c = W_clusters.get(cid, W_global)

    grp_tr  = grp[grp["year"].isin(TRAIN_YEARS)].sort_values(["year", "week"]).reset_index(drop=True)
    Z_tr    = grp_tr[STATE_COLS].values.astype(np.float32)
    Z_tr_sc = scaler.transform(Z_tr)
    exog_tr = make_exog(grp_tr) if USE_EXOG else None
    Th_p, Y_p = build_lag_features(Z_tr_sc, exog_tr)

    if Th_p is None or len(Th_p) < n_terms:
        W_provinces[prov] = W_c.copy()
    else:
        R_p    = Y_p - Th_p @ W_c
        dW_p   = stlsq(Th_p, R_p, thresh_province_vec, STLSQ_ITERS)
        W_p    = W_c + dW_p
        W_provinces[prov] = W_p

    # Evaluate val and test for this province
    for split_name, year_list in [("val", VAL_YEARS), ("test", TEST_YEARS)]:
        grp_s  = grp[grp["year"].isin(year_list)].sort_values(["year", "week"])
        grp_s  = grp_s.reset_index(drop=True)
        if len(grp_s) <= N_LAGS:
            continue
        Z_s    = grp_s[STATE_COLS].values.astype(np.float32)
        Z_s_sc = scaler.transform(Z_s)
        exog_s = make_exog(grp_s) if USE_EXOG else None
        Th_s, Y_s = build_lag_features(Z_s_sc, exog_s)
        if Th_s is None:
            continue

        Y_hat = Th_s @ W_provinces[prov]                  # (n, 7) scaled

        # Latent R2: z1..z6 columns only
        r2_lat = float(r2_score(
            Y_s[:, :LATENT_DIM].flatten(),
            Y_hat[:, :LATENT_DIM].flatten()))

        # Incidence R2: inverse-transform column 6, no decoder needed.
        # expm1 back to original scale if LOG_INCIDENCE was applied.
        Y_hat_raw = scaler.inverse_transform(Y_hat)
        Y_s_raw   = scaler.inverse_transform(Y_s)
        if LOG_INCIDENCE:
            Y_hat_raw[:, INC_COL] = np.expm1(Y_hat_raw[:, INC_COL])
            Y_s_raw[:, INC_COL]   = np.expm1(Y_s_raw[:, INC_COL])
        r2_inc    = float(r2_score(
            Y_s_raw[:, INC_COL],
            Y_hat_raw[:, INC_COL]))

        byregion.append({
            "province"     : prov,
            "cluster_id"   : cid,
            "split"        : split_name,
            "r2_latent"    : round(r2_lat, 4),
            "r2_incidence" : round(r2_inc, 4),
        })

# Aggregate level-3 results
br_df        = pd.DataFrame(byregion)
val_lat      = br_df[br_df["split"] == "val"]["r2_latent"].mean()
test_lat     = br_df[br_df["split"] == "test"]["r2_latent"].mean()
val_inc      = br_df[br_df["split"] == "val"]["r2_incidence"].mean()
test_inc     = br_df[br_df["split"] == "test"]["r2_incidence"].mean()
print(f"  Province-level  latent    val={val_lat:+.4f}  test={test_lat:+.4f}")
print(f"  Province-level  incidence val={val_inc:+.4f}  test={test_inc:+.4f}")

# ── Summary table ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EpiHSD SUMMARY")
print("=" * 60)
print(f"  {'':30s}  {'LATENT':>14s}  {'INCIDENCE':>14s}")
print(f"  {'':30s}  {'val':>6s}  {'test':>6s}  {'val':>6s}  {'test':>6s}")
print(f"  {'Level 1 Global':30s}  {r2_val_g:+.4f}  {r2_test_g:+.4f}"
      f"  {r2_val_g_inc:+.4f}  {r2_test_g_inc:+.4f}")
for row in results_lvl2:
    print(f"  {'Level 2 Cluster '+str(row['id']):30s}"
          f"  {row['val_r2_latent']:+.4f}  {row['test_r2_latent']:+.4f}"
          f"  {row['val_r2_inc']:+.4f}  {row['test_r2_inc']:+.4f}")
print(f"  {'Level 3 Province mean':30s}  {val_lat:+.4f}  {test_lat:+.4f}"
      f"  {val_inc:+.4f}  {test_inc:+.4f}")

# Per-cluster breakdown at province level
print("\n  Per-province test R2 (Level 3):")
test_prov = br_df[br_df["split"] == "test"][
    ["province", "cluster_id", "r2_latent", "r2_incidence"]]
for cid in sorted(test_prov["cluster_id"].unique()):
    rows = test_prov[test_prov["cluster_id"] == cid].sort_values(
        "r2_incidence", ascending=False)
    print(f"\n  Cluster {int(cid)}:")
    print(f"    {'Province':<35s}  {'latent':>8s}  {'incidence':>9s}")
    for _, row in rows.iterrows():
        print(f"    {row['province']:<35s}  {row['r2_latent']:+.4f}    {row['r2_incidence']:+.4f}")

# ── Save outputs ──────────────────────────────────────────────
br_df.to_csv(os.path.join(HERE, "epihsd_byregion.csv"), index=False)

summary_rows = [
    {"level": "global", "id": "all", "n_units": 32,
     "val_r2_latent" : round(r2_val_g,     4),
     "test_r2_latent": round(r2_test_g,    4),
     "val_r2_inc"    : round(r2_val_g_inc, 4),
     "test_r2_inc"   : round(r2_test_g_inc,4)}
] + results_lvl2 + [
    {"level": "province", "id": "all", "n_units": 32,
     "val_r2_latent" : round(val_lat,  4),
     "test_r2_latent": round(test_lat, 4),
     "val_r2_inc"    : round(val_inc,  4),
     "test_r2_inc"   : round(test_inc, 4)}
]
pd.DataFrame(summary_rows).to_csv(
    os.path.join(HERE, "epihsd_results.csv"), index=False)

# Save coefficient matrix for the paper
exog_names = (["time_sin", "time_cos"] + [f"rain_lag{l}" for l in range(1, N_RAIN_LAGS + 1)]
              if USE_EXOG else [])
seir_feat_names = ([f"inc_lag1_x_z{i+1}_lag1" for i in range(LATENT_DIM)] + ["inc_lag1_sq"]
                   if USE_SEIR_TERMS else [])
feat_names = ([f"lag{lag}_{col}" for lag in range(N_LAGS + 1) for col in STATE_COLS]
              + exog_names + seir_feat_names)
try:
    poly_names = poly.get_feature_names_out(feat_names)
except Exception:
    poly_names = [f"f{i}" for i in range(n_terms)]

out_names = STATE_COLS   # [z1, z2, z3, z4, z5, z6, incidence]

coef_rows = []
for d, oname in enumerate(out_names):
    for j, fname in enumerate(poly_names):
        if W_global[j, d] != 0:
            coef_rows.append({
                "level": "global", "cluster": "all",
                "output_dim": oname, "feature": fname,
                "coefficient": round(float(W_global[j, d]), 6)
            })
    for cid, W_c in W_clusters.items():
        delta = W_c - W_global
        for j, fname in enumerate(poly_names):
            if delta[j, d] != 0:
                coef_rows.append({
                    "level": "cluster_delta", "cluster": cid,
                    "output_dim": oname, "feature": fname,
                    "coefficient": round(float(delta[j, d]), 6)
                })

pd.DataFrame(coef_rows).to_csv(
    os.path.join(HERE, "epihsd_coefficients.csv"), index=False)

# Save readable equations
eq_lines = [f"EpiHSD EQUATIONS  state_dim={STATE_DIM}"
            f"  n_lags={N_LAGS}  poly_deg={POLY_DEG}\n\n"]
eq_lines.append("LEVEL 1: GLOBAL EQUATION\n")
for d, oname in enumerate(out_names):
    terms = [(poly_names[j], W_global[j, d])
             for j in range(n_terms) if W_global[j, d] != 0]
    eq = " + ".join(f"{v:.5f}*{n}" for n, v in terms) if terms else "0"
    eq_lines.append(f"  {oname}(t+1) = {eq}\n")

eq_lines.append("\nLEVEL 2: CLUSTER DELTA EQUATIONS\n")
for cid, W_c in W_clusters.items():
    delta = W_c - W_global
    eq_lines.append(f"\n  Cluster {cid}:\n")
    for d, oname in enumerate(out_names):
        terms = [(poly_names[j], delta[j, d])
                 for j in range(n_terms) if delta[j, d] != 0]
        if terms:
            eq = " + ".join(f"{v:.5f}*{n}" for n, v in terms)
            eq_lines.append(f"    delta_{oname} = {eq}\n")

with open(os.path.join(HERE, "epihsd_equations.txt"), "w") as f:
    f.writelines(eq_lines)

print(f"\n  Saved epihsd_results.csv")
print(f"  Saved epihsd_byregion.csv")
print(f"  Saved epihsd_coefficients.csv")
print(f"  Saved epihsd_equations.txt")
print("\n  DONE.")
print("─" * 60)
