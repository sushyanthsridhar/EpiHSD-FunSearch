"""
pipeline.py
────────────────────────────────────────────────────────────────
Complete end-to-end pipeline. Three stages run in sequence.

  STAGE 1  AUTOENCODER
           Train on 2015-2019. Validate on 2020-2021.
           Compress each province-year (52 weeks x 31 features)
           into 3 latent numbers z1, z2, z3.
           If a saved model exists it is loaded instead of retrained.

  STAGE 2  SINDy
           Fit sparse dynamical equations on the latent space.
           Library = polynomial z terms + annual external forcings.
           Fit on train transitions (2015-2018 -> next year).
           Validate on val transitions (2019-2020 -> next year).

  STAGE 3  FINAL EVALUATION  (test data unlocked here for first time)
           For each province in 2022-2023:
             a. Seed from actual z(2021)  (encoder output)
             b. SINDy rolls forward: z_pred(2022), z_pred(2023)
             c. Decoder reconstructs predicted 52-week incidence curves
             d. R2 and MAE computed against real weekly incidence
             e. dI/dt (week-to-week change) compared predicted vs actual

Outputs saved to autoencoder-sindy/:
  model.pt                      trained autoencoder weights
  latent_all.csv                latent coords for all 9 years
  sindy_coefficients.csv        discovered equations
  eval_metrics.csv              R2 and MAE per province per year
  plot_loss.png                 training loss curve
  plot_latent.png               3D latent scatter
  plot_incidence.png            predicted vs actual incidence curves
  plot_didt.png                 predicted vs actual dI/dt
  plot_r2_heatmap.png           test R2 per province

Run:
  python3 autoencoder-sindy/pipeline.py

Install once:
  pip install torch scikit-learn pandas numpy matplotlib
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

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
OUT_DIR = HERE
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
LATENT_DIM      = 3
WEEKS_PER_YEAR  = 52
EPOCHS          = 800
PATIENCE        = 120
HIDDEN_DIMS     = [256, 128]
PRE_BOTTLENECK  = 16
LR              = 0.00018
BATCH_SIZE      = 16
DROPOUT         = 0.09
MSE_WEIGHT      = 0.25
ACTIVATION      = "elu"

STLSQ_THRESH    = 0.05
POLY_DEG        = 2

TRAIN_YEARS     = list(range(2015, 2020))
VAL_YEARS       = [2020, 2021]
TEST_YEARS      = [2022, 2023]    # unlocked only in Stage 3

INCIDENCE_COL   = 28   # index of incidence within the 31 feature columns

# Best external forcings from mutator search
EXTERNAL_POOL = [
    "nino34_anom", "neighbor_incidence_lag1",
    "consecutive_wet_weeks", "solar_radiation",
    "soi_index", "avg_rainfall", "avg_temp",
]

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
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
    mask = np.isnan(chunks)
    chunks[mask] = np.take(col_means, np.where(mask)[1])

print(f"  Total chunks: {len(chunks)}  ({len(FEATURE_COLS)} features x {WEEKS_PER_YEAR} weeks)")

train_idx    = [i for i, l in enumerate(labels) if l["year"] in TRAIN_YEARS]
val_idx      = [i for i, l in enumerate(labels) if l["year"] in VAL_YEARS]
test_idx     = [i for i, l in enumerate(labels) if l["year"] in TEST_YEARS]
trainval_idx = train_idx + val_idx

X_train = torch.tensor(chunks[train_idx])
X_val   = torch.tensor(chunks[val_idx])
X_all   = torch.tensor(chunks)
train_loader = DataLoader(TensorDataset(X_train, X_train),
                          batch_size=BATCH_SIZE, shuffle=True)

# ═══════════════════════════════════════════════════════════════
# STAGE 1  AUTOENCODER
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  STAGE 1  AUTOENCODER")
print("═" * 60)

ACTS = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "elu": nn.ELU}

class DengueAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims,
                 pre_bottleneck, dropout, activation):
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
                enc_layers.append(act())
                enc_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*enc_layers)

        dec_dims = list(reversed(enc_dims))
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
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

def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-8))

def combined_loss(x, x_hat):
    mse = nn.functional.mse_loss(x_hat, x)
    r2  = r2_score(x, x_hat)
    return MSE_WEIGHT * mse + (1.0 - MSE_WEIGHT) * (1.0 - r2)

model = DengueAutoencoder(INPUT_DIM, LATENT_DIM, HIDDEN_DIMS,
                          PRE_BOTTLENECK, DROPOUT, ACTIVATION)
model_path = os.path.join(OUT_DIR, "model.pt")

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        val_r2  = r2_score(X_val, model(X_val))
        val_mse = nn.functional.mse_loss(model(X_val), X_val).item()
    print(f"  Loaded saved model.  val_r2={val_r2:.4f}  val_mse={val_mse:.5f}")
    train_losses, val_losses = [], []
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"  Training from scratch. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    train_losses, val_losses = [], []
    best_val_loss, patience_count = float("inf"), 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_losses = []
        for x_batch, _ in train_loader:
            optimizer.zero_grad()
            loss = combined_loss(x_batch, model(x_batch))
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = float(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            val_loss = combined_loss(X_val, model(X_val)).item()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 100 == 0 or epoch == 1:
            with torch.no_grad():
                vr2 = r2_score(X_val, model(X_val))
            print(f"  Epoch {epoch:4d}  train={train_loss:.5f}  val={val_loss:.5f}"
                  f"  val_r2={vr2:.4f}" + ("  *" if patience_count == 0 else ""))

        if patience_count >= PATIENCE:
            print(f"  Early stop at epoch {epoch}.")
            break

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        val_r2  = r2_score(X_val, model(X_val))
        val_mse = nn.functional.mse_loss(model(X_val), X_val).item()
    print(f"\n  Final  val_r2={val_r2:.4f}  val_mse={val_mse:.5f}")

# Training loss plot
if train_losses:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses,   label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Combined Loss")
    ax.set_title(f"Autoencoder Training  val_r2={val_r2:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_loss.png"), dpi=150)
    plt.close()
    print("  Saved plot_loss.png")

# Extract latent coordinates for ALL years (train + val + test)
with torch.no_grad():
    Z_all = model.encode(X_all).numpy()

z_cols = [f"z{i+1}" for i in range(LATENT_DIM)]
latent_df = pd.DataFrame(Z_all, columns=z_cols)
latent_df["province"] = [labels[i]["province"] for i in range(len(labels))]
latent_df["year"]     = [labels[i]["year"]     for i in range(len(labels))]
latent_df["split"]    = [
    "train" if labels[i]["year"] in TRAIN_YEARS
    else ("val" if labels[i]["year"] in VAL_YEARS else "test")
    for i in range(len(labels))
]
latent_df.to_csv(os.path.join(OUT_DIR, "latent_all.csv"), index=False)
print(f"  Saved latent_all.csv  ({len(latent_df)} rows)")

# 3D latent scatter
fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection="3d")
split_colors = {"train": "#457b9d", "val": "#f4a261", "test": "#e63946"}
for split, color in split_colors.items():
    m = latent_df["split"] == split
    ax.scatter(latent_df.loc[m, "z1"], latent_df.loc[m, "z2"],
               latent_df.loc[m, "z3"], c=color, s=25, alpha=0.7, label=split)
ax.set_xlabel("z1  urban scale")
ax.set_ylabel("z2  interior vs coastal")
ax.set_zlabel("z3  epidemic intensity")
ax.set_title("3D Latent Space  (all 9 years)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_latent.png"), dpi=150)
plt.close()
print("  Saved plot_latent.png")

# ═══════════════════════════════════════════════════════════════
# STAGE 2  SINDy
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  STAGE 2  SINDy")
print("═" * 60)

# Annual external features
annual = df.groupby(["province", "year"])[FEATURE_COLS].mean().reset_index()
EXTERNAL_POOL_VALID = [c for c in EXTERNAL_POOL if c in annual.columns]
print(f"  External features: {EXTERNAL_POOL_VALID}")

latent_ext = latent_df.merge(
    annual[["province", "year"] + EXTERNAL_POOL_VALID],
    on=["province", "year"], how="left"
)

# Standardize z and external features to zero mean, unit variance.
# This prevents the scale mismatch where external features (range 0-1)
# get enormous coefficients to match the unbounded z scale (range -7 to +5).
# Fit scalers on TRAIN data only, apply everywhere.
train_latent_ext = latent_ext[latent_ext["year"].isin(TRAIN_YEARS)]
z_scaler   = StandardScaler().fit(train_latent_ext[z_cols].values)
ext_scaler = StandardScaler().fit(train_latent_ext[EXTERNAL_POOL_VALID].values)

# Store training min/max for external features in original feature space.
# Used later in Stage 3 to clip OOD test values before rollout.
ext_train_min = train_latent_ext[EXTERNAL_POOL_VALID].min().values
ext_train_max = train_latent_ext[EXTERNAL_POOL_VALID].max().values

def scale_row(rows_df):
    z_s   = z_scaler.transform(rows_df[z_cols].values.astype(float))
    ext_s = ext_scaler.transform(rows_df[EXTERNAL_POOL_VALID].values.astype(float))
    return z_s, ext_s

# Build polynomial z library
poly = PolynomialFeatures(degree=POLY_DEG, include_bias=True)
# Fit poly on scaled z
z_scaled_train = z_scaler.transform(train_latent_ext[z_cols].values)
poly.fit(z_scaled_train)
z_term_names = list(poly.get_feature_names_out(z_cols))
ALL_TERMS    = z_term_names + EXTERNAL_POOL_VALID
N_TERMS      = len(ALL_TERMS)

def build_theta(rows_df):
    z_s, ext_s = scale_row(rows_df)
    Tz = poly.transform(z_s)
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
            # dZ in SCALED space so it matches the scaled Theta
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
print(f"  Train pairs: {len(df_tr)}   Val pairs: {len(df_va)}")

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

def r2_np(true, pred):
    ss_r = ((true - pred) ** 2).sum()
    ss_t = ((true - true.mean()) ** 2).sum()
    return float(1 - ss_r / (ss_t + 1e-10))

Xi = stlsq(Theta_tr, dZ_tr, STLSQ_THRESH)

print("\n  Equations (train fit):")
for j, zname in enumerate(z_cols):
    terms = [(ALL_TERMS[i], Xi[i, j]) for i in range(N_TERMS)
             if abs(Xi[i, j]) > 1e-12]
    tr_r2 = r2_np(dZ_tr[:, j], (Theta_tr @ Xi)[:, j])
    va_r2 = r2_np(dZ_va[:, j], (Theta_va @ Xi)[:, j])
    print(f"\n  d{zname}/dt  train_r2={tr_r2:.3f}  val_r2={va_r2:.3f}")
    for tname, coef in terms:
        tag = "[EXT]" if tname in EXTERNAL_POOL_VALID else "     "
        print(f"    {tag}  {coef:+.5f}  *  {tname}")
    if not terms:
        print("    0  (stable)")

coef_df = pd.DataFrame(Xi, index=ALL_TERMS,
                        columns=[f"d{z}/dt" for z in z_cols])
coef_df.to_csv(os.path.join(OUT_DIR, "sindy_coefficients.csv"))
print(f"\n  Saved sindy_coefficients.csv  (initial full library)")

# ═══════════════════════════════════════════════════════════════
# STAGE 2.5  LIBRARY MUTATION
# TRAIN data: fit Xi coefficients.
# VAL data  : compute fitness. TEST data never touched here.
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  STAGE 2.5  LIBRARY MUTATION")
print("═" * 60)

MUT_TOP_K    = 5
MUT_N_MUTS   = 15
MUT_N_ROUNDS = 30

# ── Island: fit STLSQ on a term subset, return full Xi ─────────

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
    score = 0.6*va_r2s[0] + 0.2*va_r2s[1] + 0.2*va_r2s[2] - 0.01*n_act
    return score, Xi_f, va_r2s, n_act

# ── Mutation operators ──────────────────────────────────────────

rng = np.random.default_rng(42)

def mut_residual_add(mask, Xi_f):
    inactive = np.where(~mask)[0]
    if len(inactive) == 0:
        return mask.copy()
    residuals = dZ_tr - Theta_tr @ Xi_f
    res_flat  = residuals.flatten()
    corrs = []
    for i in inactive:
        col = np.tile(Theta_tr[:, i], LATENT_DIM)
        corrs.append(abs(np.corrcoef(col, res_flat)[0, 1]))
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
    magnitudes = np.abs(Xi_f[active]).max(axis=1)
    new_mask = mask.copy()
    new_mask[active[int(np.argmin(magnitudes))]] = False
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

def mut_crossover(mask_a, mask_b):
    return mask_a | mask_b

STRAT_NAMES  = ["residual_add", "swap", "coef_prune",
                 "random_add", "random_remove", "crossover"]
STRAT_PROBS  = np.array([0.30, 0.20, 0.15, 0.15, 0.10, 0.10])

# ── Initialise elite pool with full library ─────────────────────

init_mask = np.ones(N_TERMS, dtype=bool)
init_score, init_Xi, init_va_r2s, init_n_act = fit_score(init_mask)
elite = [(init_score, init_mask.copy(), init_Xi, init_va_r2s, init_n_act)]

print(f"  Initial library  n_terms={N_TERMS}  "
      f"val_r2s={[round(v,3) for v in init_va_r2s]}  "
      f"fitness={init_score:.4f}")
print(f"  Running {MUT_N_ROUNDS} rounds x {MUT_N_MUTS} mutations ...")

best_score_history = [init_score]

for rnd in range(MUT_N_ROUNDS):
    candidates = []
    for _ in range(MUT_N_MUTS):
        parent_score, parent_mask, parent_Xi, _, _ = elite[rng.integers(len(elite))]
        sname = rng.choice(STRAT_NAMES, p=STRAT_PROBS / STRAT_PROBS.sum())

        if sname == "residual_add":
            new_mask = mut_residual_add(parent_mask, parent_Xi)
        elif sname == "swap":
            new_mask = mut_swap(parent_mask)
        elif sname == "coef_prune":
            new_mask = mut_coef_prune(parent_mask, parent_Xi)
        elif sname == "random_add":
            new_mask = mut_random_add(parent_mask)
        elif sname == "random_remove":
            new_mask = mut_random_remove(parent_mask)
        else:
            other = elite[rng.integers(len(elite))][1]
            new_mask = mut_crossover(parent_mask, other)

        score, Xi_f, va_r2s, n_act = fit_score(new_mask)
        candidates.append((score, new_mask, Xi_f, va_r2s, n_act))

    elite = sorted(elite + candidates, key=lambda x: x[0], reverse=True)[:MUT_TOP_K]
    best_score_history.append(elite[0][0])

    if (rnd + 1) % 5 == 0 or rnd == 0:
        b = elite[0]
        print(f"  Round {rnd+1:3d}  fitness={b[0]:.4f}  "
              f"val_r2s={[round(v,3) for v in b[3]]}  n_active={b[4]}")

# ── Update Xi to best library before Stage 3 ───────────────────

best_score, best_mask, Xi, best_va_r2s, best_n_act = elite[0]
active_terms = [ALL_TERMS[i] for i in range(N_TERMS) if best_mask[i]]

print(f"\n  Best library  n_active={best_n_act}  fitness={best_score:.4f}")
print(f"  val_r2s z1={best_va_r2s[0]:.4f}  z2={best_va_r2s[1]:.4f}  z3={best_va_r2s[2]:.4f}")
print("  Active terms:")
for t in active_terms:
    print(f"    {t}")

coef_df_best = pd.DataFrame(Xi, index=ALL_TERMS,
                             columns=[f"d{z}/dt" for z in z_cols])
coef_df_best.to_csv(os.path.join(OUT_DIR, "sindy_coefficients.csv"))
print("  Updated sindy_coefficients.csv with best library.")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(best_score_history, color="#457b9d", lw=2)
ax.set_xlabel("Round")
ax.set_ylabel("Best fitness")
ax.set_title("Library Mutation Convergence")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_mutation_convergence.png"), dpi=150)
plt.close()
print("  Saved plot_mutation_convergence.png")

# ═══════════════════════════════════════════════════════════════
# STAGE 3  FINAL EVALUATION (test data unlocked)
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  STAGE 3  FINAL EVALUATION  (2022-2023 test data)")
print("═" * 60)

# For each test province:
#   1. Seed = actual z(2021) from encoder
#   2. SINDy rollout -> z_pred(2022), z_pred(2023)
#   3. Decoder -> predicted 52-week incidence curve
#   4. R2 on weekly incidence and on dI/dt against real data

eval_records = []
provinces = sorted(latent_ext["province"].unique())

# Get actual 52-week incidence for test years (normalized)
def get_incidence_curve(province, year):
    rows = df[(df["province"] == province) & (df["year"] == year)]
    rows = rows.sort_values("week")
    if len(rows) < WEEKS_PER_YEAR:
        return None
    return rows[FEATURE_COLS[INCIDENCE_COL]].values[:WEEKS_PER_YEAR].astype(float)

actual_incidence_col = FEATURE_COLS[INCIDENCE_COL]
print(f"  Incidence column: '{actual_incidence_col}' (index {INCIDENCE_COL})")

# Diagnostic: show 2022 external values relative to training range.
test_2022_ext = latent_ext[latent_ext["year"] == 2022][EXTERNAL_POOL_VALID]
if len(test_2022_ext) > 0:
    print("\n  External feature clipping diagnostics (2022 mean vs training range):")
    for fi, feat in enumerate(EXTERNAL_POOL_VALID):
        v    = test_2022_ext[feat].mean()
        lo   = ext_train_min[fi]
        hi   = ext_train_max[fi]
        flag = " <CLIPPED>" if (v < lo or v > hi) else ""
        print(f"    {feat:30s}  val={v:+.4f}  train=[{lo:+.4f}, {hi:+.4f}]{flag}")

for province in provinces:
    sub = latent_ext[latent_ext["province"] == province].sort_values("year")

    seed_row = sub[sub["year"] == 2021]
    if seed_row.empty:
        continue
    # z_cur is in ORIGINAL (unscaled) z space — decoder expects this
    z_cur = seed_row[z_cols].values[0].astype(float)

    for test_year in TEST_YEARS:
        test_row = sub[sub["year"] == test_year]
        if test_row.empty:
            continue

        # Step 1: SINDy predict dz.
        # build_theta expects original z — it scales internally.
        row_df   = pd.DataFrame([test_row.iloc[0]])
        row_df_z = row_df.copy()
        for k, zc in enumerate(z_cols):
            row_df_z[zc] = z_cur[k]

        # Clip external features to training distribution range.
        # 2022 climate values (nino34_anom, soi_index) are 3-4 std devs
        # outside training range and cause rollout divergence without clipping.
        for fi, feat in enumerate(EXTERNAL_POOL_VALID):
            row_df_z[feat] = float(np.clip(
                float(row_df_z[feat].values[0]),
                ext_train_min[fi],
                ext_train_max[fi]
            ))

        t_row = build_theta(row_df_z)
        # dz is in SCALED space. Un-scale to get actual z step.
        dz_scaled = (t_row @ Xi).flatten()
        dz_actual = dz_scaled * z_scaler.scale_   # un-standardize step
        z_pred    = z_cur + dz_actual
        z_cur     = z_pred.copy()   # seed for next year in original z space

        # Step 2: Decode z_pred (original z space) -> full 52-week curve
        with torch.no_grad():
            z_tensor = torch.tensor(z_pred.astype(np.float32)).unsqueeze(0)
            decoded  = model.decode(z_tensor).numpy().flatten()  # (1612,)

        # Step 3: Extract incidence from decoded output
        decoded_2d     = decoded.reshape(WEEKS_PER_YEAR, len(FEATURE_COLS))
        pred_incidence = decoded_2d[:, INCIDENCE_COL]   # (52,)

        # Step 4: Get actual incidence
        actual_incidence = get_incidence_curve(province, test_year)
        if actual_incidence is None:
            continue

        # Step 5: Compute dI/dt (week-to-week change)
        pred_didt   = np.diff(pred_incidence)    # (51,)
        actual_didt = np.diff(actual_incidence)  # (51,)

        # Step 6: Metrics
        r2_inc  = r2_np(actual_incidence, pred_incidence)
        mae_inc = float(np.abs(actual_incidence - pred_incidence).mean())
        r2_didt = r2_np(actual_didt, pred_didt)
        mae_didt= float(np.abs(actual_didt - pred_didt).mean())

        eval_records.append({
            "province":  province,
            "year":      test_year,
            "r2_incidence":  r2_inc,
            "mae_incidence": mae_inc,
            "r2_didt":       r2_didt,
            "mae_didt":      mae_didt,
            "pred_incidence":  pred_incidence.tolist(),
            "actual_incidence": actual_incidence.tolist(),
            "pred_didt":   pred_didt.tolist(),
            "actual_didt": actual_didt.tolist(),
            "z_pred":      z_pred.tolist(),
            "z_actual":    test_row[z_cols].values[0].tolist(),
        })

eval_df = pd.DataFrame(eval_records)

# Summary metrics
print("\n  Summary across all test provinces:")
print(f"  Incidence  R2 : mean={eval_df['r2_incidence'].mean():.4f}  "
      f"median={eval_df['r2_incidence'].median():.4f}  "
      f"min={eval_df['r2_incidence'].min():.4f}")
print(f"  Incidence  MAE: mean={eval_df['mae_incidence'].mean():.4f}")
print(f"  dI/dt      R2 : mean={eval_df['r2_didt'].mean():.4f}  "
      f"median={eval_df['r2_didt'].median():.4f}  "
      f"min={eval_df['r2_didt'].min():.4f}")
print(f"  dI/dt      MAE: mean={eval_df['mae_didt'].mean():.4f}")

print("\n  Best 5 provinces (incidence R2):")
best5 = eval_df.groupby("province")["r2_incidence"].mean().sort_values(ascending=False).head(5)
print(best5.round(3).to_string())
print("\n  Worst 5 provinces (incidence R2):")
worst5 = eval_df.groupby("province")["r2_incidence"].mean().sort_values().head(5)
print(worst5.round(3).to_string())

# Save metrics (without list columns)
metrics_save = eval_df.drop(columns=["pred_incidence","actual_incidence",
                                      "pred_didt","actual_didt",
                                      "z_pred","z_actual"])
metrics_save.to_csv(os.path.join(OUT_DIR, "eval_metrics.csv"), index=False)
print(f"\n  Saved eval_metrics.csv")

# ── Plot 1: Incidence curves (8 sample provinces) ───────────────
sample_provs = sorted(provinces)[:8]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
weeks = np.arange(1, WEEKS_PER_YEAR + 1)

for pi, province in enumerate(sample_provs):
    ax  = axes[pi]
    sub = eval_df[eval_df["province"] == province].sort_values("year")
    colors = {2022: ("#457b9d", "#e63946"), 2023: ("#2a9d8f", "#f4a261")}
    for _, row in sub.iterrows():
        yr = row["year"]
        ac, pc = colors.get(yr, ("#888", "#ccc"))
        ax.plot(weeks, row["actual_incidence"], "-",  color=ac, lw=2,
                alpha=0.9, label=f"{yr} actual")
        ax.plot(weeks, row["pred_incidence"],   "--", color=pc, lw=2,
                alpha=0.9, label=f"{yr} pred")
    mean_r2 = sub["r2_incidence"].mean()
    ax.set_title(f"{province.split(' ',1)[1][:14]}  R2={mean_r2:.2f}", fontsize=9)
    ax.set_xlabel("Week", fontsize=8)
    ax.set_ylabel("Incidence (norm.)", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)
    if pi == 0:
        ax.legend(fontsize=6, ncol=2)

plt.suptitle("Predicted vs Actual Weekly Incidence  (Test 2022-2023)\n"
             "solid=actual  dashed=SINDy+decoder prediction",
             fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_incidence.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plot_incidence.png")

# ── Plot 2: dI/dt curves ────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
dweeks = np.arange(1, WEEKS_PER_YEAR)

for pi, province in enumerate(sample_provs):
    ax  = axes[pi]
    sub = eval_df[eval_df["province"] == province].sort_values("year")
    colors = {2022: ("#457b9d", "#e63946"), 2023: ("#2a9d8f", "#f4a261")}
    for _, row in sub.iterrows():
        yr = row["year"]
        ac, pc = colors.get(yr, ("#888", "#ccc"))
        ax.plot(dweeks, row["actual_didt"], "-",  color=ac, lw=1.5,
                alpha=0.9, label=f"{yr} actual")
        ax.plot(dweeks, row["pred_didt"],   "--", color=pc, lw=1.5,
                alpha=0.9, label=f"{yr} pred")
    ax.axhline(0, color="gray", lw=0.8, linestyle=":")
    mean_r2 = sub["r2_didt"].mean()
    ax.set_title(f"{province.split(' ',1)[1][:14]}  R2={mean_r2:.2f}", fontsize=9)
    ax.set_xlabel("Week", fontsize=8)
    ax.set_ylabel("dI/dt (norm.)", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)
    if pi == 0:
        ax.legend(fontsize=6, ncol=2)

plt.suptitle("Predicted vs Actual  dI/dt  (Test 2022-2023)\n"
             "solid=actual  dashed=SINDy+decoder  dotted line=zero",
             fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_didt.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plot_didt.png")

# ── Plot 3: R2 heatmap per province ─────────────────────────────
prov_r2 = eval_df.groupby("province")[["r2_incidence", "r2_didt"]].mean()
fig, ax = plt.subplots(figsize=(6, max(7, len(prov_r2) * 0.32)))
im = ax.imshow(prov_r2.values, aspect="auto",
               cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_xticks([0, 1])
ax.set_xticklabels(["R2 incidence", "R2 dI/dt"])
ax.set_yticks(range(len(prov_r2)))
ax.set_yticklabels([p.split(" ", 1)[1][:22] for p in prov_r2.index], fontsize=7)
for i in range(len(prov_r2)):
    for j in range(2):
        val = prov_r2.values[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=6.5, color="black")
plt.colorbar(im, ax=ax, label="R2")
ax.set_title("Test R2 per Province  (mean 2022+2023)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_r2_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved plot_r2_heatmap.png")

print("\n" + "═" * 60)
print("  PIPELINE COMPLETE")
print("═" * 60)
print(f"  Incidence R2  mean = {eval_df['r2_incidence'].mean():.4f}")
print(f"  dI/dt     R2  mean = {eval_df['r2_didt'].mean():.4f}")
print(f"\n  All outputs in: autoencoder-sindy/")
print("─" * 60)
