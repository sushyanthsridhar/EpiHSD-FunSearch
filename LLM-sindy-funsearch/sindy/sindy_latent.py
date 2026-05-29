"""
sindy_latent.py
────────────────────────────────────────────────────────────────
SINDy on the autoencoder latent space.

PIPELINE:
  1. Load latent_representations_latentN.csv  (train+val only)
  2. Compute dz/dt via forward finite difference between years
  3. Build polynomial library Theta(z) up to degree POLY_DEG
  4. Fit sparse coefficients via STLSQ
  5. Print discovered equations clearly
  6. If latent_all_years_latentN.csv exists: evaluate on test years
     (run train_autoencoder_eval_latent3.py first to generate it)

KEY KNOB: THRESHOLD
  Too low  -> dense, overfitted equations
  Too high -> everything zeroed out
  Start at 0.05, try 0.01 and 0.10 to see sensitivity.

Run:
  python3 sindy_latent.py

Install once:
  pip install scikit-learn pandas numpy matplotlib
────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ── Config ─────────────────────────────────────────────────────
LATENT_DIM  = 3       # change to match the autoencoder you ran
POLY_DEG    = 3       # degree of polynomial library
THRESHOLD   = 0.15    # STLSQ sparsity threshold  <-- main dial
MAX_ITER    = 20

TRAIN_YEARS = list(range(2015, 2020))
VAL_YEARS   = [2020, 2021]
TEST_YEARS  = [2022, 2023]

# ── Paths ──────────────────────────────────────────────────────
LATENT_CSV     = os.path.join(ROOT, "autoencoder",
                               f"latent_representations_latent{LATENT_DIM}.csv")
ALL_YEARS_CSV  = os.path.join(ROOT, "autoencoder",
                               f"latent_all_years_latent{LATENT_DIM}.csv")
OUT_DIR = HERE
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load train+val latent coords ───────────────────────────────
df     = pd.read_csv(LATENT_CSV)
z_cols = [f"z{i+1}" for i in range(LATENT_DIM)]

assert df["year"].max() <= 2021, "TEST DATA LEAK. Stop."
print(f"Loaded {len(df)} rows. Years: {sorted(df.year.unique())}")
print(f"Provinces: {df.province.nunique()}")

# ── Build (state, derivative) pairs ───────────────────────────
# Each consecutive-year pair per province gives one training row.
# dz/dt ~ z(t+1) - z(t),  dt = 1 year.
Z_list, dZ_list = [], []

for province, grp in df.groupby("province"):
    grp = grp.sort_values("year").reset_index(drop=True)
    for t in range(len(grp) - 1):
        if grp.loc[t + 1, "year"] - grp.loc[t, "year"] == 1:
            Z_list.append(grp.loc[t, z_cols].values.astype(float))
            dZ_list.append(
                grp.loc[t + 1, z_cols].values.astype(float)
                - grp.loc[t, z_cols].values.astype(float)
            )

Z_data  = np.array(Z_list,  dtype=np.float64)
dZ_data = np.array(dZ_list, dtype=np.float64)
print(f"State-derivative pairs for fitting: {len(Z_data)}")

# ── Polynomial candidate library ──────────────────────────────
poly  = PolynomialFeatures(degree=POLY_DEG, include_bias=True)
Theta = poly.fit_transform(Z_data)
names = poly.get_feature_names_out(z_cols)
print(f"Library: {len(names)} terms")

# ── STLSQ ─────────────────────────────────────────────────────
def stlsq(Theta, dZ, threshold, max_iter):
    D  = dZ.shape[1]
    Xi = np.linalg.lstsq(Theta, dZ, rcond=None)[0]
    for _ in range(max_iter):
        prev = Xi.copy()
        for j in range(D):
            active = np.abs(Xi[:, j]) >= threshold
            Xi[:, j] = 0.0
            if active.sum() > 0:
                Xi[active, j] = np.linalg.lstsq(
                    Theta[:, active], dZ[:, j], rcond=None)[0]
        if np.allclose(Xi, prev, atol=1e-12):
            break
    return Xi

Xi = stlsq(Theta, dZ_data, THRESHOLD, MAX_ITER)

# ── Print discovered equations ─────────────────────────────────
print("\n" + "=" * 65)
print(f"  DISCOVERED EQUATIONS   (THRESHOLD = {THRESHOLD})")
print("=" * 65)
for j, zname in enumerate(z_cols):
    terms = [(names[i], Xi[i, j])
             for i in range(len(names)) if abs(Xi[i, j]) > 1e-12]
    print(f"\n  d{zname}/dt =")
    if terms:
        for tname, coef in terms:
            print(f"      {coef:+.5f}  *  {tname}")
    else:
        print("      0   [no terms survived threshold]")
print("\n" + "=" * 65)

# ── Fit quality on train+val ───────────────────────────────────
dZ_pred = Theta @ Xi
print("\nFit quality (train+val):")
for j, zname in enumerate(z_cols):
    ss_res = ((dZ_data[:, j] - dZ_pred[:, j]) ** 2).sum()
    ss_tot = ((dZ_data[:, j] - dZ_data[:, j].mean()) ** 2).sum()
    r2     = 1.0 - ss_res / (ss_tot + 1e-10)
    n_act  = (np.abs(Xi[:, j]) > 1e-12).sum()
    print(f"  d{zname}/dt   R2 = {r2:.4f}   active terms = {n_act}/{len(names)}")

# ── Save coefficient table ─────────────────────────────────────
coef_df = pd.DataFrame(Xi, index=names,
                        columns=[f"d{z}/dt" for z in z_cols])
coef_path = os.path.join(OUT_DIR, f"sindy_coefficients_latent{LATENT_DIM}.csv")
coef_df.to_csv(coef_path)
print(f"\nSaved {coef_path}")

# ── Residual plots (train+val) ─────────────────────────────────
fig, axes = plt.subplots(1, LATENT_DIM, figsize=(5 * LATENT_DIM, 4))
if LATENT_DIM == 1:
    axes = [axes]
for j, zname in enumerate(z_cols):
    ax = axes[j]
    ax.scatter(dZ_data[:, j], dZ_pred[:, j], alpha=0.6, s=25, color="#457b9d")
    lim = max(np.abs(dZ_data[:, j]).max(), np.abs(dZ_pred[:, j]).max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
    ss_res = ((dZ_data[:, j] - dZ_pred[:, j]) ** 2).sum()
    ss_tot = ((dZ_data[:, j] - dZ_data[:, j].mean()) ** 2).sum()
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)
    ax.set_title(f"d{zname}/dt  R2={r2:.3f}  (train+val)")
    ax.set_xlabel(f"True d{zname}/dt")
    ax.set_ylabel(f"Predicted d{zname}/dt")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"sindy_residuals_latent{LATENT_DIM}.png"), dpi=150)
plt.close()
print(f"Saved sindy_residuals_latent{LATENT_DIM}.png")

# ══════════════════════════════════════════════════════════════
# TEST EVALUATION
# Requires latent_all_years_latentN.csv.
# Generate it by running: train_autoencoder_eval_latent3.py
# ══════════════════════════════════════════════════════════════
if not os.path.exists(ALL_YEARS_CSV):
    print(f"\n[TEST EVAL] {ALL_YEARS_CSV} not found.")
    print("  Run train_autoencoder_eval_latent3.py first, then re-run this script.")
    print("\n── DONE ─────────────────────────────────────────────────")
    exit()

print(f"\n[TEST EVAL] Loading {ALL_YEARS_CSV}")
df_all = pd.read_csv(ALL_YEARS_CSV)

# ── Predict test years by rolling forward from last val state ──
# For each province:
#   z_pred(2022) = z_actual(2021) + f(z_actual(2021))
#   z_pred(2023) = z_pred(2022)   + f(z_pred(2022))
# Compare to z_actual(2022), z_actual(2023).

provinces  = sorted(df_all["province"].unique())
all_actual = []
all_pred   = []
prov_labels = []

for province in provinces:
    sub = df_all[df_all["province"] == province].sort_values("year")

    # seed = last known val state
    seed_row = sub[sub["year"] == 2021]
    if seed_row.empty:
        continue

    z_seed = seed_row[z_cols].values[0]   # shape (D,)
    test_rows = sub[sub["year"].isin(TEST_YEARS)].sort_values("year")
    if test_rows.empty:
        continue

    # roll forward
    z_current = z_seed.copy()
    preds = []
    for _ in range(len(test_rows)):
        theta_row = poly.transform(z_current.reshape(1, -1))
        dz        = (theta_row @ Xi).flatten()
        z_current = z_current + dz
        preds.append(z_current.copy())

    preds  = np.array(preds)                        # (n_test_years, D)
    actual = test_rows[z_cols].values               # (n_test_years, D)

    all_actual.append(actual)
    all_pred.append(preds)
    prov_labels.extend([province] * len(test_rows))

all_actual = np.vstack(all_actual)   # (n_provinces * n_test_years, D)
all_pred   = np.vstack(all_pred)

print("\nTest performance (SINDy rollout 2022-2023):")
for j, zname in enumerate(z_cols):
    ss_res = ((all_actual[:, j] - all_pred[:, j]) ** 2).sum()
    ss_tot = ((all_actual[:, j] - all_actual[:, j].mean()) ** 2).sum()
    r2     = 1.0 - ss_res / (ss_tot + 1e-10)
    mae    = np.abs(all_actual[:, j] - all_pred[:, j]).mean()
    print(f"  {zname}   R2 = {r2:.4f}   MAE = {mae:.4f}")

# ── Test residual scatter ──────────────────────────────────────
fig, axes = plt.subplots(1, LATENT_DIM, figsize=(5 * LATENT_DIM, 4))
if LATENT_DIM == 1:
    axes = [axes]
for j, zname in enumerate(z_cols):
    ax = axes[j]
    ax.scatter(all_actual[:, j], all_pred[:, j],
               alpha=0.7, s=35, color="#e63946")
    lim = max(np.abs(all_actual[:, j]).max(), np.abs(all_pred[:, j]).max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ss_res = ((all_actual[:, j] - all_pred[:, j]) ** 2).sum()
    ss_tot = ((all_actual[:, j] - all_actual[:, j].mean()) ** 2).sum()
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)
    ax.set_title(f"{zname}   R2={r2:.3f}  (TEST 2022-2023)")
    ax.set_xlabel(f"Actual {zname}")
    ax.set_ylabel(f"SINDy predicted {zname}")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"sindy_test_scatter_latent{LATENT_DIM}.png"), dpi=150)
plt.close()
print(f"Saved sindy_test_scatter_latent{LATENT_DIM}.png")

# ── Trajectory plot: 6 sample provinces ───────────────────────
# Shows actual vs SINDy-predicted latent trajectory over all years.
sample_provs = provinces[:6]
fig, axes = plt.subplots(LATENT_DIM, len(sample_provs),
                          figsize=(4 * len(sample_provs), 3 * LATENT_DIM))

for pi, province in enumerate(sample_provs):
    sub    = df_all[df_all["province"] == province].sort_values("year")
    years  = sub["year"].tolist()

    seed_row  = sub[sub["year"] == 2021]
    z_current = seed_row[z_cols].values[0]
    rollout   = {2021: z_current.copy()}
    for yr in [2022, 2023]:
        theta_row = poly.transform(z_current.reshape(1, -1))
        dz        = (theta_row @ Xi).flatten()
        z_current = z_current + dz
        rollout[yr] = z_current.copy()

    for j, zname in enumerate(z_cols):
        ax = axes[j, pi] if LATENT_DIM > 1 else axes[pi]
        ax.plot(years, sub[zname].values, "o-", color="#457b9d",
                lw=2, ms=5, label="Actual")
        pred_years = [2021, 2022, 2023]
        pred_vals  = [rollout[y][j] for y in pred_years]
        ax.plot(pred_years, pred_vals, "s--", color="#e63946",
                lw=2, ms=5, label="SINDy" if pi == 0 else "")
        ax.axvline(2021.5, color="gray", lw=1, linestyle=":")
        ax.set_title(province.split(" ", 1)[1][:12], fontsize=8)
        if pi == 0:
            ax.set_ylabel(zname)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", rotation=45, labelsize=7)

axes[0, 0].legend(fontsize=7) if LATENT_DIM > 1 else axes[0].legend(fontsize=7)
plt.suptitle(
    f"SINDy Latent Trajectory Rollout  (dashed = predicted, dotted line = train/test split)",
    fontsize=10
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"sindy_trajectories_latent{LATENT_DIM}.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved sindy_trajectories_latent{LATENT_DIM}.png")
print("\n── DONE ─────────────────────────────────────────────────")
