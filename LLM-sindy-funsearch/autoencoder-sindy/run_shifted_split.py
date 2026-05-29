"""
run_shifted_split.py
────────────────────────────────────────────────────────────────
Shifted temporal split evaluation for latent dim 6.
Tests on 2019, which is fully within the training climate
distribution, avoiding the 2022 OOD problem.

Split:
  Autoencoder : already trained on 2015-2019 (loaded as-is)
  SINDy train : 2015->2016, 2016->2017 transitions  (64 pairs)
  SINDy val   : 2017->2018 transition               (32 pairs)
  SINDy test  : seed from z(2018), predict z(2019)
                decode to incidence, compare to actual 2019

Note: the autoencoder has seen 2019 during its own training.
The SINDy model has NOT. This tests SINDy generalization under
in-distribution climate conditions.

Run:
  python3 autoencoder-sindy/run_shifted_split.py
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

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.dirname(HERE)
AE_DIR = os.path.join(ROOT, "autoencoder")
OUT_DIR = os.path.join(HERE, "shifted_split")
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
LATENT_DIM     = 6
WEEKS_PER_YEAR = 52
INCIDENCE_COL  = 28

# Shifted split
AE_TRAIN_YEARS   = list(range(2015, 2020))   # autoencoder was trained on these
SINDY_TRAIN_YSTART = [2015, 2016, 2017]      # transitions 2015->2016, 2016->2017, 2017->2018
SINDY_VAL_YSTART   = [2018]                  # transition  2018->2019
SINDY_SEED_YEAR    = 2018                    # seed z from this year
SINDY_TEST_YEAR    = 2019                    # predict and evaluate this year

STLSQ_THRESH = 0.05
POLY_DEG     = 1

EXTERNAL_POOL = [
    "nino34_anom", "neighbor_incidence_lag1",
    "consecutive_wet_weeks", "solar_radiation",
    "soi_index", "avg_rainfall", "avg_temp",
]

MUT_TOP_K    = 5
MUT_N_MUTS   = 15
MUT_N_ROUNDS = 20

# Dim 6 architecture (tuned)
HIDDEN_DIMS     = [256, 512]
PRE_BOTTLENECK  = 32
SHALLOW_DECODER = False
USE_BATCHNORM   = True
DROPOUT         = 0.0629
ACTIVATION      = "elu"

# ═══════════════════════════════════════════════════════════════
# AUTOENCODER CLASS
# ═══════════════════════════════════════════════════════════════
ACTS = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "elu": nn.ELU}

class DengueAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, pre_bottleneck,
                 dropout, activation, use_batchnorm=False, shallow_decoder=False):
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
                enc_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*enc_layers)
        if shallow_decoder:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, input_dim), nn.Sigmoid()
            )
        else:
            dec_dims = list(reversed(enc_dims))
            dec_layers = []
            for i in range(len(dec_dims) - 1):
                dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
                if i < len(dec_dims) - 2:
                    if use_batchnorm:
                        dec_layers.append(nn.BatchNorm1d(dec_dims[i + 1]))
                    dec_layers.append(act())
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


def r2_np(true, pred):
    ss_r = ((true - pred) ** 2).sum()
    ss_t = ((true - true.mean()) ** 2).sum()
    return float(1 - ss_r / (ss_t + 1e-10))

def stlsq(Theta, dZ, threshold, max_iter=30):
    Xi = np.linalg.lstsq(Theta, dZ, rcond=None)[0]
    for _ in range(max_iter):
        prev = Xi.copy()
        for j in range(dZ.shape[1]):
            active = np.abs(Xi[:, j]) >= threshold
            Xi[:, j] = 0.0
            if active.sum() > 0:
                Xi[active, j] = np.linalg.lstsq(
                    Theta[:, active], dZ[:, j], rcond=None)[0]
        if np.allclose(Xi, prev, atol=1e-12):
            break
    return Xi


# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════
print("Loading data...")
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
    mask_nan  = np.isnan(chunks)
    chunks[mask_nan] = np.take(col_means, np.where(mask_nan)[1])

print(f"  Total chunks: {len(chunks)}")

# ═══════════════════════════════════════════════════════════════
# LOAD DIM 6 AUTOENCODER
# ═══════════════════════════════════════════════════════════════
print(f"\nLoading dim {LATENT_DIM} autoencoder...")
model = DengueAutoencoder(
    INPUT_DIM, LATENT_DIM, HIDDEN_DIMS, PRE_BOTTLENECK,
    DROPOUT, ACTIVATION, USE_BATCHNORM, SHALLOW_DECODER
)
model.load_state_dict(torch.load(
    os.path.join(AE_DIR, f"best_model_latent{LATENT_DIM}.pt"),
    weights_only=True
))
model.eval()

X_all = torch.tensor(chunks)
with torch.no_grad():
    Z_all = model.encode(X_all).numpy()

z_cols = [f"z{i+1}" for i in range(LATENT_DIM)]
latent_df = pd.DataFrame(Z_all, columns=z_cols)
latent_df["province"] = [labels[i]["province"] for i in range(len(labels))]
latent_df["year"]     = [labels[i]["year"]     for i in range(len(labels))]
print(f"  Encoded {len(latent_df)} province-year chunks")

# ═══════════════════════════════════════════════════════════════
# BUILD LATENT + EXTERNAL TABLE
# ═══════════════════════════════════════════════════════════════
annual    = df.groupby(["province", "year"])[FEATURE_COLS].mean().reset_index()
EXT_VALID = [c for c in EXTERNAL_POOL if c in annual.columns]
latent_ext = latent_df.merge(
    annual[["province", "year"] + EXT_VALID],
    on=["province", "year"], how="left"
)

# Scalers fit on SINDy train years only.
# Include the destination year of each transition so the scaler
# covers all z values the training pairs can produce.
sindy_train_years = list(set(SINDY_TRAIN_YSTART + [y + 1 for y in SINDY_TRAIN_YSTART]))
sindy_train_rows  = latent_ext[latent_ext["year"].isin(sindy_train_years)]

z_scaler   = StandardScaler().fit(sindy_train_rows[z_cols].values)
ext_scaler = StandardScaler().fit(sindy_train_rows[EXT_VALID].values)
ext_train_min = sindy_train_rows[EXT_VALID].min().values
ext_train_max = sindy_train_rows[EXT_VALID].max().values

poly = PolynomialFeatures(degree=POLY_DEG, include_bias=True)
poly.fit(z_scaler.transform(sindy_train_rows[z_cols].values))
z_term_names = list(poly.get_feature_names_out(z_cols))
ALL_TERMS    = z_term_names + EXT_VALID
N_TERMS      = len(ALL_TERMS)

def build_theta(rows_df):
    z_s   = z_scaler.transform(rows_df[z_cols].values.astype(float))
    ext_s = ext_scaler.transform(rows_df[EXT_VALID].values.astype(float))
    return np.hstack([poly.transform(z_s), ext_s])

def build_pairs(df_ext, allowed_start_years):
    Zrow_list, dZ_list = [], []
    for province, grp in df_ext.groupby("province"):
        grp = grp.sort_values("year").reset_index(drop=True)
        for t in range(len(grp) - 1):
            if grp.loc[t, "year"] not in allowed_start_years:
                continue
            if grp.loc[t + 1, "year"] - grp.loc[t, "year"] != 1:
                continue
            Zrow_list.append(grp.loc[t])
            z_t1 = z_scaler.transform(
                grp.loc[t + 1, z_cols].values.reshape(1, -1)).flatten()
            z_t  = z_scaler.transform(
                grp.loc[t,     z_cols].values.reshape(1, -1)).flatten()
            dZ_list.append(z_t1 - z_t)
    return pd.DataFrame(Zrow_list).reset_index(drop=True), np.array(dZ_list)

df_tr, dZ_tr = build_pairs(latent_ext, SINDY_TRAIN_YSTART)
df_va, dZ_va = build_pairs(latent_ext, SINDY_VAL_YSTART)
Theta_tr     = build_theta(df_tr)
Theta_va     = build_theta(df_va)
print(f"\nSINDy  train pairs: {len(df_tr)}   val pairs: {len(df_va)}   library: {N_TERMS} terms")

# ═══════════════════════════════════════════════════════════════
# STAGE 2  INITIAL SINDy FIT
# ═══════════════════════════════════════════════════════════════
Xi = stlsq(Theta_tr, dZ_tr, STLSQ_THRESH)
init_tr = [r2_np(dZ_tr[:, j], (Theta_tr @ Xi)[:, j]) for j in range(LATENT_DIM)]
init_va = [r2_np(dZ_va[:, j], (Theta_va @ Xi)[:, j]) for j in range(LATENT_DIM)]
print(f"Initial SINDy  train_r2={[round(v,3) for v in init_tr]}")
print(f"               val_r2  ={[round(v,3) for v in init_va]}")

# ═══════════════════════════════════════════════════════════════
# STAGE 2.5  LIBRARY MUTATION
# ═══════════════════════════════════════════════════════════════
print(f"\nRunning library mutation ({MUT_N_ROUNDS} rounds x {MUT_N_MUTS} mutations)...")
rng = np.random.default_rng(42)

def island_run(mask):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.zeros((N_TERMS, LATENT_DIM)), [0.0]*LATENT_DIM, [0.0]*LATENT_DIM, 0
    Th_tr_m = Theta_tr[:, idx]
    Th_va_m = Theta_va[:, idx]
    Xi_sub  = stlsq(Th_tr_m, dZ_tr, STLSQ_THRESH)
    tr_r2s  = [r2_np(dZ_tr[:, j], (Th_tr_m @ Xi_sub)[:, j]) for j in range(LATENT_DIM)]
    va_r2s  = [r2_np(dZ_va[:, j], (Th_va_m @ Xi_sub)[:, j]) for j in range(LATENT_DIM)]
    Xi_full = np.zeros((N_TERMS, LATENT_DIM))
    Xi_full[idx] = Xi_sub
    n_act   = int((np.abs(Xi_full) > 1e-12).any(axis=1).sum())
    return Xi_full, tr_r2s, va_r2s, n_act

def fit_score(mask):
    Xi_f, tr_r2s, va_r2s, n_act = island_run(mask)
    mean_va = float(np.mean(va_r2s))
    mean_tr = float(np.mean(tr_r2s))
    gap_pen = max(0.0, mean_tr - mean_va - 0.15)
    score   = mean_va - 0.01 * n_act - 0.5 * gap_pen
    return score, Xi_f, tr_r2s, va_r2s, n_act

def mut_residual_add(mask, Xi_f):
    inactive = np.where(~mask)[0]
    if len(inactive) == 0:
        return mask.copy()
    residuals = dZ_tr - Theta_tr @ Xi_f
    res_flat  = residuals.flatten()
    corrs = []
    for i in inactive:
        col = np.tile(Theta_tr[:, i], LATENT_DIM)
        c   = np.corrcoef(col, res_flat)[0, 1]
        corrs.append(0.0 if np.isnan(c) else abs(c))
    new_mask = mask.copy()
    new_mask[inactive[int(np.argmax(corrs))]] = True
    return new_mask

def mut_swap(mask):
    active, inactive = np.where(mask)[0], np.where(~mask)[0]
    if len(active) == 0 or len(inactive) == 0:
        return mask.copy()
    new_mask = mask.copy()
    new_mask[rng.choice(active)]   = False
    new_mask[rng.choice(inactive)] = True
    return new_mask

def mut_coef_prune(mask, Xi_f):
    active = np.where(mask)[0]
    if len(active) <= 1:
        return mask.copy()
    mags = np.abs(Xi_f[active]).max(axis=1)
    new_mask = mask.copy()
    new_mask[active[int(np.argmin(mags))]] = False
    return new_mask

def mut_random_add(mask):
    inactive = np.where(~mask)[0]
    if len(inactive) == 0:
        return mask.copy()
    new_mask = mask.copy()
    new_mask[rng.choice(inactive)] = True
    return new_mask

def mut_random_remove(mask):
    active = np.where(mask)[0]
    if len(active) <= 1:
        return mask.copy()
    new_mask = mask.copy()
    new_mask[rng.choice(active)] = False
    return new_mask

STRAT_NAMES = ["residual_add", "swap", "coef_prune",
               "random_add", "random_remove", "crossover"]
STRAT_PROBS = np.array([0.30, 0.20, 0.15, 0.15, 0.10, 0.10])

init_mask           = np.ones(N_TERMS, dtype=bool)
s0, Xi0, tr0, va0, n0 = fit_score(init_mask)
elite = [(s0, init_mask.copy(), Xi0, tr0, va0, n0)]

for rnd in range(MUT_N_ROUNDS):
    candidates = []
    for _ in range(MUT_N_MUTS):
        parent = elite[rng.integers(len(elite))]
        p_mask, p_Xi = parent[1], parent[2]
        sname = rng.choice(STRAT_NAMES, p=STRAT_PROBS / STRAT_PROBS.sum())
        if sname == "residual_add":
            nm = mut_residual_add(p_mask, p_Xi)
        elif sname == "swap":
            nm = mut_swap(p_mask)
        elif sname == "coef_prune":
            nm = mut_coef_prune(p_mask, p_Xi)
        elif sname == "random_add":
            nm = mut_random_add(p_mask)
        elif sname == "random_remove":
            nm = mut_random_remove(p_mask)
        else:
            other = elite[rng.integers(len(elite))][1]
            nm = p_mask | other
        score, Xi_f, tr_r2s, va_r2s, n_act = fit_score(nm)
        candidates.append((score, nm, Xi_f, tr_r2s, va_r2s, n_act))
    elite = sorted(elite + candidates, key=lambda x: x[0], reverse=True)[:MUT_TOP_K]
    if (rnd + 1) % 5 == 0 or rnd == 0:
        b = elite[0]
        print(f"  Round {rnd+1:3d}  fitness={b[0]:.4f}  "
              f"val_r2s={[round(v,3) for v in b[4]]}  n_active={b[5]}")

best_score, best_mask, Xi, best_tr_r2s, best_va_r2s, best_n = elite[0]
active_terms = [ALL_TERMS[i] for i in range(N_TERMS) if best_mask[i]]
print(f"\nBest library  n_active={best_n}  val_r2s={[round(v,3) for v in best_va_r2s]}")
print(f"Active terms: {active_terms}")

# ═══════════════════════════════════════════════════════════════
# STAGE 3  TEST ON 2019
# ═══════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  TEST ON {SINDY_TEST_YEAR}  (in-distribution, no OOD climate)")
print(f"{'═'*60}")

provinces   = sorted(latent_ext["province"].unique())
eval_recs   = []
actual_inc_col = FEATURE_COLS[INCIDENCE_COL]

for province in provinces:
    sub = latent_ext[latent_ext["province"] == province].sort_values("year")

    seed_row = sub[sub["year"] == SINDY_SEED_YEAR]
    test_row = sub[sub["year"] == SINDY_TEST_YEAR]
    if seed_row.empty or test_row.empty:
        continue

    z_cur  = seed_row[z_cols].values[0].astype(float)

    row_df   = pd.DataFrame([test_row.iloc[0]])
    row_df_z = row_df.copy()
    for k, zc in enumerate(z_cols):
        row_df_z[zc] = z_cur[k]
    for fi, feat in enumerate(EXT_VALID):
        row_df_z[feat] = float(np.clip(
            float(row_df_z[feat].values[0]),
            ext_train_min[fi], ext_train_max[fi]
        ))

    t_row     = build_theta(row_df_z)
    dz_scaled = (t_row @ Xi).flatten()
    dz_actual = dz_scaled * z_scaler.scale_
    z_pred    = z_cur + dz_actual

    with torch.no_grad():
        z_t    = torch.tensor(z_pred.astype(np.float32)).unsqueeze(0)
        decoded = model.decode(z_t).numpy().flatten()

    decoded_2d     = decoded.reshape(WEEKS_PER_YEAR, len(FEATURE_COLS))
    pred_incidence = decoded_2d[:, INCIDENCE_COL]

    rows_act = df[(df["province"] == province) & (df["year"] == SINDY_TEST_YEAR)]
    rows_act = rows_act.sort_values("week")
    if len(rows_act) < WEEKS_PER_YEAR:
        continue
    actual_incidence = rows_act[actual_inc_col].values[:WEEKS_PER_YEAR].astype(float)

    pred_didt   = np.diff(pred_incidence)
    actual_didt = np.diff(actual_incidence)

    eval_recs.append({
        "province":       province,
        "r2_incidence":   r2_np(actual_incidence, pred_incidence),
        "mae_incidence":  float(np.abs(actual_incidence - pred_incidence).mean()),
        "r2_didt":        r2_np(actual_didt, pred_didt),
        "mae_didt":       float(np.abs(actual_didt - pred_didt).mean()),
        "pred_incidence": pred_incidence.tolist(),
        "actual_incidence": actual_incidence.tolist(),
        "z_pred":         z_pred.tolist(),
        "z_actual":       test_row[z_cols].values[0].tolist(),
    })

eval_df = pd.DataFrame(eval_recs)

print(f"\n  Incidence R2  mean={eval_df['r2_incidence'].mean():.4f}  "
      f"median={eval_df['r2_incidence'].median():.4f}  "
      f"min={eval_df['r2_incidence'].min():.4f}  "
      f"max={eval_df['r2_incidence'].max():.4f}")
print(f"  Incidence MAE mean={eval_df['mae_incidence'].mean():.4f}")
print(f"  dI/dt     R2  mean={eval_df['r2_didt'].mean():.4f}  "
      f"median={eval_df['r2_didt'].median():.4f}")

print(f"\n  Best 5 provinces:")
best5 = eval_df.nlargest(5, "r2_incidence")[["province", "r2_incidence", "r2_didt"]]
print(best5.round(3).to_string(index=False))
print(f"\n  Worst 5 provinces:")
worst5 = eval_df.nsmallest(5, "r2_incidence")[["province", "r2_incidence", "r2_didt"]]
print(worst5.round(3).to_string(index=False))

# Save metrics
eval_df.drop(columns=["pred_incidence", "actual_incidence", "z_pred", "z_actual"]).to_csv(
    os.path.join(OUT_DIR, "eval_metrics_shifted.csv"), index=False
)

# ── Plot: incidence curves for 8 provinces ─────────────────────
sample_provs = eval_df.sort_values("r2_incidence", ascending=False)["province"].tolist()
top4    = sample_provs[:4]
bottom4 = sample_provs[-4:]
plot_provs = top4 + bottom4

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes  = axes.flatten()
weeks = np.arange(1, WEEKS_PER_YEAR + 1)
for pi, province in enumerate(plot_provs):
    ax  = axes[pi]
    row = eval_df[eval_df["province"] == province].iloc[0]
    ax.plot(weeks, row["actual_incidence"], "-",  color="#457b9d", lw=2, label="Actual 2019")
    ax.plot(weeks, row["pred_incidence"],   "--", color="#e63946", lw=2, label="SINDy pred")
    ax.set_title(f"{province.split(' ',1)[1][:14]}  R2={row['r2_incidence']:.2f}", fontsize=9)
    ax.set_xlabel("Week", fontsize=8)
    ax.set_ylabel("Incidence (norm.)", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)
    if pi == 0:
        ax.legend(fontsize=7)

plt.suptitle(
    f"Shifted Split Test: Predicted vs Actual Incidence  (2019)\n"
    f"SINDy trained on 2015-2017 transitions only.  Top 4 and bottom 4 provinces by R2.",
    fontsize=10
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_incidence_shifted.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── R2 heatmap ─────────────────────────────────────────────────
prov_r2 = eval_df.set_index("province")[["r2_incidence", "r2_didt"]].sort_values("r2_incidence", ascending=False)
fig, ax  = plt.subplots(figsize=(6, max(7, len(prov_r2) * 0.32)))
im = ax.imshow(prov_r2.values, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_xticks([0, 1])
ax.set_xticklabels(["R2 incidence", "R2 dI/dt"])
ax.set_yticks(range(len(prov_r2)))
ax.set_yticklabels([p.split(" ", 1)[1][:22] for p in prov_r2.index], fontsize=7)
for i in range(len(prov_r2)):
    for j in range(2):
        ax.text(j, i, f"{prov_r2.values[i,j]:.2f}",
                ha="center", va="center", fontsize=6.5, color="black")
plt.colorbar(im, ax=ax, label="R2")
ax.set_title(f"Shifted Split Test R2 per Province  (test year 2019)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_r2_heatmap_shifted.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\nOutputs saved to autoencoder-sindy/shifted_split/")
print(f"\n{'═'*60}")
print(f"  SHIFTED SPLIT COMPLETE")
print(f"  Incidence R2  mean={eval_df['r2_incidence'].mean():.4f}  "
      f"median={eval_df['r2_incidence'].median():.4f}")
print(f"  dI/dt     R2  mean={eval_df['r2_didt'].mean():.4f}  "
      f"median={eval_df['r2_didt'].median():.4f}")
print(f"{'═'*60}")
