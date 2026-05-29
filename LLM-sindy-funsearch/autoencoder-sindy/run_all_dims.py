"""
run_all_dims.py
────────────────────────────────────────────────────────────────
Runs the full pipeline (autoencoder load, SINDy, library mutation,
test evaluation) for latent dimensions 2 through 6, using the
pre-trained models in ../autoencoder/best_model_latentN.pt.

Outputs per dim are saved to autoencoder-sindy/dim_N/.
A comparison table is printed at the end.

Run:
  python3 autoencoder-sindy/run_all_dims.py
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
from torch.utils.data import DataLoader, TensorDataset

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(HERE)
AE_DIR = os.path.join(ROOT, "autoencoder")

# ═══════════════════════════════════════════════════════════════
# SHARED CONFIG  (same across all dims)
# ═══════════════════════════════════════════════════════════════
WEEKS_PER_YEAR = 52
STLSQ_THRESH   = 0.05
POLY_DEG       = 2
INCIDENCE_COL  = 28

TRAIN_YEARS    = list(range(2015, 2020))
VAL_YEARS      = [2020, 2021]
TEST_YEARS     = [2022, 2023]

EXTERNAL_POOL  = [
    "nino34_anom", "neighbor_incidence_lag1",
    "consecutive_wet_weeks", "solar_radiation",
    "soi_index", "avg_rainfall", "avg_temp",
]

MUT_TOP_K    = 5
MUT_N_MUTS   = 15
MUT_N_ROUNDS = 20

# ── Per-dim architecture configs ────────────────────────────────
# Dims 2-6: tuned hyperparams from Optuna.
# Dims 7-10: placeholder using dim 6 config until tuning is done.
#            After running tune_hyperparams_latentN.py and
#            train_autoencoder_latentN.py, update these entries
#            with the best config from best_config_latentN.txt.
DIM_CONFIGS = {
    2:  dict(hidden_dims=[512, 128, 1024], pre_bottleneck=4,
             shallow_decoder=True,  use_batchnorm=False,
             dropout=0.2135, activation="relu", mse_weight=0.3359),
    3:  dict(hidden_dims=[256, 128],       pre_bottleneck=16,
             shallow_decoder=False, use_batchnorm=False,
             dropout=0.09,   activation="elu",  mse_weight=0.25),
    4:  dict(hidden_dims=[256],            pre_bottleneck=8,
             shallow_decoder=True,  use_batchnorm=True,
             dropout=0.2739, activation="relu", mse_weight=0.1432),
    5:  dict(hidden_dims=[256, 512],       pre_bottleneck=64,
             shallow_decoder=False, use_batchnorm=True,
             dropout=0.3871, activation="elu",  mse_weight=0.8333),
    6:  dict(hidden_dims=[256, 512],       pre_bottleneck=32,
             shallow_decoder=False, use_batchnorm=True,
             dropout=0.0629, activation="elu",  mse_weight=0.5298),
    # ── placeholder configs for dims 7-10 (update after tuning) ──
    7:  dict(hidden_dims=[256, 512],       pre_bottleneck=64,
             shallow_decoder=False, use_batchnorm=True,
             dropout=0.21,   activation="relu", mse_weight=0.41),
    8:  dict(hidden_dims=[256, 512],       pre_bottleneck=32,
             shallow_decoder=False, use_batchnorm=True,
             dropout=0.0629, activation="elu",  mse_weight=0.5298),
    9:  dict(hidden_dims=[256, 512],       pre_bottleneck=32,
             shallow_decoder=False, use_batchnorm=True,
             dropout=0.0629, activation="elu",  mse_weight=0.5298),
    10: dict(hidden_dims=[256, 512],       pre_bottleneck=32,
             shallow_decoder=False, use_batchnorm=True,
             dropout=0.0629, activation="elu",  mse_weight=0.5298),
}

ACTS = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "elu": nn.ELU}

# ═══════════════════════════════════════════════════════════════
# AUTOENCODER CLASS  (unified, covers all dim configs)
# ═══════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════
def r2_score_torch(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-8))

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
# LOAD DATA  (done once, shared across all dims)
# ═══════════════════════════════════════════════════════════════
print("Loading data...")
df = pd.read_csv(os.path.join(ROOT, "extended_input_normalized.csv"))
ID_COLS      = ["province", "year", "month", "week"]
FEATURE_COLS = [c for c in df.columns if c not in ID_COLS]
INPUT_DIM    = WEEKS_PER_YEAR * len(FEATURE_COLS)
print(f"  Features per week: {len(FEATURE_COLS)}   Input dim: {INPUT_DIM}")

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

train_idx = [i for i, l in enumerate(labels) if l["year"] in TRAIN_YEARS]
val_idx   = [i for i, l in enumerate(labels) if l["year"] in VAL_YEARS]
X_train   = torch.tensor(chunks[train_idx])
X_val     = torch.tensor(chunks[val_idx])
X_all     = torch.tensor(chunks)

annual = df.groupby(["province", "year"])[FEATURE_COLS].mean().reset_index()
EXT_VALID = [c for c in EXTERNAL_POOL if c in annual.columns]

actual_incidence_col = FEATURE_COLS[INCIDENCE_COL]
print(f"  Incidence column: '{actual_incidence_col}' (index {INCIDENCE_COL})")


# ═══════════════════════════════════════════════════════════════
# PER-DIM PIPELINE FUNCTION
# ═══════════════════════════════════════════════════════════════
def run_dim(latent_dim):
    cfg     = DIM_CONFIGS[latent_dim]
    out_dir = os.path.join(HERE, f"dim_{latent_dim}")
    os.makedirs(out_dir, exist_ok=True)

    z_cols = [f"z{i+1}" for i in range(latent_dim)]

    print(f"\n{'═'*60}")
    print(f"  LATENT DIM = {latent_dim}")
    print(f"{'═'*60}")

    # ── Load pre-trained model ──────────────────────────────────
    model_path = os.path.join(AE_DIR, f"best_model_latent{latent_dim}.pt")
    model = DengueAutoencoder(
        INPUT_DIM, latent_dim,
        cfg["hidden_dims"], cfg["pre_bottleneck"],
        cfg["dropout"], cfg["activation"],
        cfg.get("use_batchnorm", False),
        cfg.get("shallow_decoder", False),
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        val_r2  = r2_score_torch(X_val, model(X_val))
        val_mse = nn.functional.mse_loss(model(X_val), X_val).item()
    print(f"  Autoencoder  val_r2={val_r2:.4f}  val_mse={val_mse:.5f}")

    # ── Extract latent coords for all years ────────────────────
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
    latent_df.to_csv(os.path.join(out_dir, "latent_all.csv"), index=False)

    # ── Build latent + external feature table ──────────────────
    latent_ext = latent_df.merge(
        annual[["province", "year"] + EXT_VALID],
        on=["province", "year"], how="left"
    )

    train_le = latent_ext[latent_ext["year"].isin(TRAIN_YEARS)]
    z_scaler   = StandardScaler().fit(train_le[z_cols].values)
    ext_scaler = StandardScaler().fit(train_le[EXT_VALID].values)
    ext_train_min = train_le[EXT_VALID].min().values
    ext_train_max = train_le[EXT_VALID].max().values

    # ── Build poly library ─────────────────────────────────────
    poly = PolynomialFeatures(degree=POLY_DEG, include_bias=True)
    poly.fit(z_scaler.transform(train_le[z_cols].values))
    z_term_names = list(poly.get_feature_names_out(z_cols))
    ALL_TERMS    = z_term_names + EXT_VALID
    N_TERMS      = len(ALL_TERMS)

    def build_theta(rows_df):
        z_s   = z_scaler.transform(rows_df[z_cols].values.astype(float))
        ext_s = ext_scaler.transform(rows_df[EXT_VALID].values.astype(float))
        Tz    = poly.transform(z_s)
        return np.hstack([Tz, ext_s])

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
        df_rows = pd.DataFrame(Zrow_list).reset_index(drop=True)
        return df_rows, np.array(dZ_list)

    df_tr, dZ_tr = build_pairs(latent_ext, [2015, 2016, 2017, 2018])
    df_va, dZ_va = build_pairs(latent_ext, [2019, 2020])
    Theta_tr     = build_theta(df_tr)
    Theta_va     = build_theta(df_va)
    print(f"  Train pairs: {len(df_tr)}   Val pairs: {len(df_va)}   Library: {N_TERMS} terms")

    # ── Initial SINDy fit ──────────────────────────────────────
    Xi = stlsq(Theta_tr, dZ_tr, STLSQ_THRESH)
    init_tr_r2s = [r2_np(dZ_tr[:, j], (Theta_tr @ Xi)[:, j]) for j in range(latent_dim)]
    init_va_r2s = [r2_np(dZ_va[:, j], (Theta_va @ Xi)[:, j]) for j in range(latent_dim)]
    print(f"  Initial SINDy  train_r2={[round(v,3) for v in init_tr_r2s]}  "
          f"val_r2={[round(v,3) for v in init_va_r2s]}")

    # ── Library mutation ───────────────────────────────────────
    rng = np.random.default_rng(42)

    def island_run(mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return np.zeros((N_TERMS, latent_dim)), [0.0]*latent_dim, [0.0]*latent_dim, 0
        Th_tr_m = Theta_tr[:, idx]
        Th_va_m = Theta_va[:, idx]
        Xi_sub  = stlsq(Th_tr_m, dZ_tr, STLSQ_THRESH)
        tr_r2s  = [r2_np(dZ_tr[:, j], (Th_tr_m @ Xi_sub)[:, j]) for j in range(latent_dim)]
        va_r2s  = [r2_np(dZ_va[:, j], (Th_va_m @ Xi_sub)[:, j]) for j in range(latent_dim)]
        Xi_full = np.zeros((N_TERMS, latent_dim))
        Xi_full[idx] = Xi_sub
        n_act   = int((np.abs(Xi_full) > 1e-12).any(axis=1).sum())
        return Xi_full, tr_r2s, va_r2s, n_act

    def fitness(va_r2s, n_act, tr_r2s):
        mean_va  = float(np.mean(va_r2s))
        mean_tr  = float(np.mean(tr_r2s))
        gap_pen  = max(0.0, mean_tr - mean_va - 0.15)
        return mean_va - 0.01 * n_act - 0.5 * gap_pen

    def fit_score(mask):
        Xi_f, tr_r2s, va_r2s, n_act = island_run(mask)
        score = fitness(va_r2s, n_act, tr_r2s)
        return score, Xi_f, tr_r2s, va_r2s, n_act

    def mut_residual_add(mask, Xi_f):
        inactive = np.where(~mask)[0]
        if len(inactive) == 0:
            return mask.copy()
        residuals = dZ_tr - Theta_tr @ Xi_f
        res_flat  = residuals.flatten()
        corrs = []
        for i in inactive:
            col = np.tile(Theta_tr[:, i], latent_dim)
            c   = np.corrcoef(col, res_flat)[0, 1]
            corrs.append(0.0 if np.isnan(c) else abs(c))
        new_mask = mask.copy()
        new_mask[inactive[int(np.argmax(corrs))]] = True
        return new_mask

    def mut_swap(mask):
        active   = np.where(mask)[0]
        inactive = np.where(~mask)[0]
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

    init_mask   = np.ones(N_TERMS, dtype=bool)
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
            print(f"  Mut round {rnd+1:3d}  fitness={b[0]:.4f}  "
                  f"val_r2s={[round(v,3) for v in b[4]]}  n_active={b[5]}")

    best_score, best_mask, Xi, best_tr_r2s, best_va_r2s, best_n_act = elite[0]
    active_terms = [ALL_TERMS[i] for i in range(N_TERMS) if best_mask[i]]
    print(f"  Best library  n_active={best_n_act}  "
          f"val_r2s={[round(v,3) for v in best_va_r2s]}")
    print(f"  Terms: {active_terms}")

    # ── Test evaluation ────────────────────────────────────────
    provinces  = sorted(latent_ext["province"].unique())
    eval_recs  = []

    for province in provinces:
        sub = latent_ext[latent_ext["province"] == province].sort_values("year")
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
            for fi, feat in enumerate(EXT_VALID):
                row_df_z[feat] = float(np.clip(
                    float(row_df_z[feat].values[0]),
                    ext_train_min[fi], ext_train_max[fi]
                ))

            t_row     = build_theta(row_df_z)
            dz_scaled = (t_row @ Xi).flatten()
            dz_actual = dz_scaled * z_scaler.scale_
            z_pred    = z_cur + dz_actual
            z_cur     = z_pred.copy()

            with torch.no_grad():
                z_t    = torch.tensor(z_pred.astype(np.float32)).unsqueeze(0)
                decoded = model.decode(z_t).numpy().flatten()

            decoded_2d     = decoded.reshape(WEEKS_PER_YEAR, len(FEATURE_COLS))
            pred_incidence = decoded_2d[:, INCIDENCE_COL]

            rows_act = df[(df["province"] == province) & (df["year"] == test_year)]
            rows_act = rows_act.sort_values("week")
            if len(rows_act) < WEEKS_PER_YEAR:
                continue
            actual_incidence = rows_act[actual_incidence_col].values[:WEEKS_PER_YEAR].astype(float)

            pred_didt   = np.diff(pred_incidence)
            actual_didt = np.diff(actual_incidence)

            eval_recs.append({
                "province":      province,
                "year":          test_year,
                "r2_incidence":  r2_np(actual_incidence, pred_incidence),
                "mae_incidence": float(np.abs(actual_incidence - pred_incidence).mean()),
                "r2_didt":       r2_np(actual_didt, pred_didt),
                "mae_didt":      float(np.abs(actual_didt - pred_didt).mean()),
            })

    eval_df = pd.DataFrame(eval_recs)
    eval_df.to_csv(os.path.join(out_dir, "eval_metrics.csv"), index=False)

    print(f"\n  ── TEST RESULTS  dim={latent_dim} ──")
    print(f"  Incidence R2  mean={eval_df['r2_incidence'].mean():.4f}  "
          f"median={eval_df['r2_incidence'].median():.4f}")
    print(f"  dI/dt     R2  mean={eval_df['r2_didt'].mean():.4f}  "
          f"median={eval_df['r2_didt'].median():.4f}")
    print(f"  Best province:  {eval_df.groupby('province')['r2_incidence'].mean().idxmax()}  "
          f"R2={eval_df.groupby('province')['r2_incidence'].mean().max():.3f}")

    # ── R2 heatmap ─────────────────────────────────────────────
    prov_r2 = eval_df.groupby("province")[["r2_incidence", "r2_didt"]].mean()
    fig, ax  = plt.subplots(figsize=(6, max(7, len(prov_r2) * 0.32)))
    im = ax.imshow(prov_r2.values, aspect="auto",
                   cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["R2 incidence", "R2 dI/dt"])
    ax.set_yticks(range(len(prov_r2)))
    ax.set_yticklabels([p.split(" ", 1)[1][:22] for p in prov_r2.index], fontsize=7)
    for i in range(len(prov_r2)):
        for j in range(2):
            ax.text(j, i, f"{prov_r2.values[i,j]:.2f}",
                    ha="center", va="center", fontsize=6.5, color="black")
    plt.colorbar(im, ax=ax, label="R2")
    ax.set_title(f"Test R2 per Province  latent_dim={latent_dim}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_r2_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "dim":          latent_dim,
        "ae_val_r2":    round(val_r2, 4),
        "sindy_best_lib_terms": best_n_act,
        "sindy_val_r2_mean":    round(float(np.mean(best_va_r2s)), 4),
        "test_inc_r2_mean":     round(eval_df["r2_incidence"].mean(), 4),
        "test_inc_r2_median":   round(eval_df["r2_incidence"].median(), 4),
        "test_didt_r2_mean":    round(eval_df["r2_didt"].mean(), 4),
        "test_didt_r2_median":  round(eval_df["r2_didt"].median(), 4),
        "test_inc_r2_best":     round(eval_df.groupby("province")["r2_incidence"].mean().max(), 4),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN  run all dims and compare
# ═══════════════════════════════════════════════════════════════
summary_rows = []
for dim in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    row = run_dim(dim)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).set_index("dim")
summary_df.to_csv(os.path.join(HERE, "dim_comparison.csv"))

print("\n\n" + "═" * 70)
print("  COMPARISON ACROSS ALL LATENT DIMENSIONS")
print("═" * 70)
print(summary_df.to_string())
print("═" * 70)
print(f"\nSaved dim_comparison.csv and per-dim results in autoencoder-sindy/dim_N/")
