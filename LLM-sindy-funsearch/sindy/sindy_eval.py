"""
sindy_eval.py
────────────────────────────────────────────────────────────────
End-to-end SINDy evaluation pipeline. Three explicit phases:

  PHASE 1  TRAIN   fit SINDy on 2015-2019 transitions only
  PHASE 2  VAL     one-step prediction on 2019-2021 transitions
  PHASE 3  TEST    rollout prediction on 2022-2023 (unseen data)

Uses the best library found by sindy_mutator.py by default.
You can override LIBRARY_TERMS below to try any combination.

Outputs (all saved to sindy/):
  sindy_eval_metrics.csv          R2 and MAE for all three phases
  sindy_eval_residuals.png        scatter: predicted vs actual dz/dt
  sindy_eval_trajectories.png     per-province latent path + rollout
  sindy_eval_test_heatmap.png     test MAE per province per z axis

Run:
  python3 sindy/sindy_eval.py
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

# ── Config ──────────────────────────────────────────────────────
LATENT_DIM   = 3
POLY_DEG     = 2
STLSQ_THRESH = 0.05

TRAIN_YEARS = list(range(2015, 2020))
VAL_YEARS   = [2020, 2021]
TEST_YEARS  = [2022, 2023]

z_cols = [f"z{i+1}" for i in range(LATENT_DIM)]

# Best library from mutator. Edit freely to experiment.
# Set to None to use all polynomial z terms only (no external).
LIBRARY_TERMS = [
    # z polynomial terms
    "1", "z1", "z1^2", "z1 z3",
    # external forcings discovered by mutator
    "nino34_anom", "neighbor_incidence_lag1",
    "consecutive_wet_weeks", "solar_radiation",
]

# ── Load data ───────────────────────────────────────────────────
latent_tv  = pd.read_csv(os.path.join(ROOT, "autoencoder",
                          f"latent_representations_latent{LATENT_DIM}.csv"))
latent_all = pd.read_csv(os.path.join(ROOT, "autoencoder",
                          f"latent_all_years_latent{LATENT_DIM}.csv"))

ext     = pd.read_csv(os.path.join(ROOT, "extended_input_normalized.csv"))
id_cols = ["province", "year", "month", "week"]
feat_cols = [c for c in ext.columns if c not in id_cols]
annual  = ext.groupby(["province", "year"])[feat_cols].mean().reset_index()

# Identify which LIBRARY_TERMS are external (not polynomial z)
poly_base = PolynomialFeatures(degree=POLY_DEG, include_bias=True)
poly_base.fit(np.zeros((1, LATENT_DIM)))
z_term_names = list(poly_base.get_feature_names_out(z_cols))

if LIBRARY_TERMS is None:
    LIBRARY_TERMS = z_term_names

ext_terms = [t for t in LIBRARY_TERMS if t not in z_term_names]
print(f"Library: {len(LIBRARY_TERMS)} terms")
print(f"  Z polynomial : {[t for t in LIBRARY_TERMS if t in z_term_names]}")
print(f"  External     : {ext_terms}")

def merge_ext(latent_df):
    if not ext_terms:
        return latent_df
    return latent_df.merge(
        annual[["province", "year"] + ext_terms],
        on=["province", "year"], how="left"
    )

tv_df  = merge_ext(latent_tv)
all_df = merge_ext(latent_all)

# ── Build Theta matrix for a dataframe ─────────────────────────
poly_base.fit(tv_df[z_cols].values)

def make_theta(df_rows):
    """Build feature matrix from a set of rows (one per transition)."""
    Z = df_rows[z_cols].values.astype(float)
    Tz = poly_base.transform(Z)
    name_to_col = {n: Tz[:, i] for i, n in enumerate(z_term_names)}

    cols = []
    col_names = []
    for term in LIBRARY_TERMS:
        if term in name_to_col:
            cols.append(name_to_col[term])
        elif term in df_rows.columns:
            cols.append(df_rows[term].values.astype(float))
        else:
            raise ValueError(f"Term '{term}' not found. Check LIBRARY_TERMS.")
        col_names.append(term)
    return np.column_stack(cols), col_names

# ── Build transition pairs ───────────────────────────────────────
def build_pairs(df, allowed_start_years):
    rows_t, rows_t1 = [], []
    for province, grp in df.groupby("province"):
        grp = grp.sort_values("year").reset_index(drop=True)
        for t in range(len(grp) - 1):
            if grp.loc[t, "year"] not in allowed_start_years:
                continue
            if grp.loc[t + 1, "year"] - grp.loc[t, "year"] != 1:
                continue
            rows_t.append(grp.loc[t])
            rows_t1.append(grp.loc[t + 1])
    df_t  = pd.DataFrame(rows_t).reset_index(drop=True)
    df_t1 = pd.DataFrame(rows_t1).reset_index(drop=True)
    dZ    = (df_t1[z_cols].values - df_t[z_cols].values).astype(float)
    return df_t, dZ

# TRAIN: transitions starting in 2015-2018 (arrive in 2016-2019)
df_tr, dZ_tr = build_pairs(tv_df,  allowed_start_years=[2015,2016,2017,2018])
# VAL:   transitions starting in 2019-2020 (arrive in 2020-2021)
df_va, dZ_va = build_pairs(tv_df,  allowed_start_years=[2019, 2020])
print(f"\nTrain pairs : {len(df_tr)}")
print(f"Val pairs   : {len(df_va)}")

Theta_tr, term_names = make_theta(df_tr)
Theta_va, _          = make_theta(df_va)

# ── STLSQ ───────────────────────────────────────────────────────
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

def r2(true, pred):
    ss_r = ((true - pred) ** 2).sum()
    ss_t = ((true - true.mean()) ** 2).sum()
    return float(1 - ss_r / (ss_t + 1e-10))

# ── PHASE 1: TRAIN ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 1  TRAIN  (2015-2019)")
print("=" * 65)
Xi = stlsq(Theta_tr, dZ_tr, STLSQ_THRESH)
dZ_tr_pred = Theta_tr @ Xi

train_metrics = {}
for j, zname in enumerate(z_cols):
    r2_j  = r2(dZ_tr[:, j], dZ_tr_pred[:, j])
    mae_j = float(np.abs(dZ_tr[:, j] - dZ_tr_pred[:, j]).mean())
    n_act = int((np.abs(Xi[:, j]) > 1e-12).sum())
    train_metrics[zname] = {"r2": r2_j, "mae": mae_j, "n_active": n_act}
    print(f"  d{zname}/dt   R2={r2_j:.4f}   MAE={mae_j:.4f}   active={n_act}/{len(term_names)}")

print("\n  Discovered equations:")
for j, zname in enumerate(z_cols):
    terms = [(term_names[i], Xi[i, j]) for i in range(len(term_names))
             if abs(Xi[i, j]) > 1e-12]
    print(f"\n  d{zname}/dt =")
    for tname, coef in terms:
        tag = "[EXT]" if tname in ext_terms else "     "
        print(f"    {tag}  {coef:+.5f}  *  {tname}")
    if not terms:
        print("    0  (stable)")

# ── PHASE 2: VAL ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 2  VAL  one-step prediction  (2020-2021)")
print("=" * 65)
dZ_va_pred = Theta_va @ Xi
val_metrics = {}
for j, zname in enumerate(z_cols):
    r2_j  = r2(dZ_va[:, j], dZ_va_pred[:, j])
    mae_j = float(np.abs(dZ_va[:, j] - dZ_va_pred[:, j]).mean())
    val_metrics[zname] = {"r2": r2_j, "mae": mae_j}
    print(f"  d{zname}/dt   R2={r2_j:.4f}   MAE={mae_j:.4f}")

# ── PHASE 3: TEST ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PHASE 3  TEST  rollout prediction  (2022-2023, unseen)")
print("=" * 65)
all_actual, all_pred, test_records = [], [], []

for province in sorted(all_df["province"].unique()):
    sub  = all_df[all_df["province"] == province].sort_values("year")
    seed = sub[sub["year"] == 2021]
    if seed.empty:
        continue
    z_cur = seed[z_cols].values[0].astype(float)
    test  = sub[sub["year"].isin(TEST_YEARS)].sort_values("year")
    if test.empty:
        continue

    preds = []
    for _, row in test.iterrows():
        row_df    = pd.DataFrame([row])
        t_row, _  = make_theta(row_df)
        # substitute current predicted z into z columns of t_row
        # (re-evaluate polynomial part from current z_cur)
        row_df_pred = row_df.copy()
        for k, zc in enumerate(z_cols):
            row_df_pred[zc] = z_cur[k]
        t_row_pred, _ = make_theta(row_df_pred)
        dz    = (t_row_pred @ Xi).flatten()
        z_cur = z_cur + dz
        preds.append(z_cur.copy())

    preds  = np.array(preds)
    actual = test[z_cols].values.astype(float)
    all_actual.append(actual)
    all_pred.append(preds)

    for yr_i, yr in enumerate(test["year"].tolist()):
        rec = {"province": province, "year": yr}
        for k, zname in enumerate(z_cols):
            rec[f"actual_{zname}"] = float(actual[yr_i, k])
            rec[f"pred_{zname}"]   = float(preds[yr_i, k])
            rec[f"err_{zname}"]    = float(abs(actual[yr_i, k] - preds[yr_i, k]))
        test_records.append(rec)

all_actual = np.vstack(all_actual)
all_pred   = np.vstack(all_pred)
test_df    = pd.DataFrame(test_records)
test_metrics = {}

for j, zname in enumerate(z_cols):
    r2_j  = r2(all_actual[:, j], all_pred[:, j])
    mae_j = float(np.abs(all_actual[:, j] - all_pred[:, j]).mean())
    test_metrics[zname] = {"r2": r2_j, "mae": mae_j}
    print(f"  {zname}   R2={r2_j:.4f}   MAE={mae_j:.4f}")

print("\n  Worst 5 provinces (z1 test MAE):")
prov_z1_mae = test_df.groupby("province")["err_z1"].mean().sort_values(ascending=False)
print(prov_z1_mae.head(5).round(3).to_string())
print("\n  Best 5 provinces (z1 test MAE):")
print(prov_z1_mae.tail(5).round(3).to_string())

# ── Save metrics ─────────────────────────────────────────────────
rows = []
for phase, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
    for zname, m in metrics.items():
        rows.append({"phase": phase, "axis": zname, **m})
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(os.path.join(HERE, f"sindy_eval_metrics_latent{LATENT_DIM}.csv"), index=False)
print(f"\nSaved sindy_eval_metrics_latent{LATENT_DIM}.csv")

# ── Plot 1: Residual scatter (train + val side by side) ──────────
fig, axes = plt.subplots(2, LATENT_DIM, figsize=(5 * LATENT_DIM, 8))
colors = {"train": "#457b9d", "val": "#2a9d8f"}
for phase_idx, (label, dZ_true, dZ_hat) in enumerate([
    ("train", dZ_tr, dZ_tr_pred),
    ("val",   dZ_va, dZ_va_pred),
]):
    for j, zname in enumerate(z_cols):
        ax = axes[phase_idx, j]
        ax.scatter(dZ_true[:, j], dZ_hat[:, j],
                   alpha=0.6, s=25, color=colors[label])
        lim = max(np.abs(dZ_true[:, j]).max(), np.abs(dZ_hat[:, j]).max()) * 1.15
        ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
        r2_j = r2(dZ_true[:, j], dZ_hat[:, j])
        ax.set_title(f"{label.upper()}  d{zname}/dt  R2={r2_j:.3f}")
        ax.set_xlabel(f"True d{zname}/dt")
        ax.set_ylabel(f"Predicted")
        ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(HERE, f"sindy_eval_residuals_latent{LATENT_DIM}.png"), dpi=150)
plt.close()

# ── Plot 2: Trajectory rollout for 8 sample provinces ───────────
sample_provs = sorted(all_df["province"].unique())[:8]
fig, axes = plt.subplots(LATENT_DIM, len(sample_provs),
                          figsize=(4 * len(sample_provs), 3.5 * LATENT_DIM))

for pi, province in enumerate(sample_provs):
    sub = all_df[all_df["province"] == province].sort_values("year")
    prov_test = test_df[test_df["province"] == province].sort_values("year")

    for j, zname in enumerate(z_cols):
        ax = axes[j, pi]
        # Full actual trajectory
        ax.plot(sub["year"], sub[zname], "o-", color="#457b9d",
                lw=2, ms=4, label="Actual")
        # SINDy rollout from 2021
        pred_years = [2021] + prov_test["year"].tolist()
        pred_vals  = [float(sub[sub["year"] == 2021][zname].values[0])]
        pred_vals += prov_test[f"pred_{zname}"].tolist()
        ax.plot(pred_years, pred_vals, "s--", color="#e63946",
                lw=2, ms=4, label="SINDy" if pi == 0 else "")
        ax.axvline(2021.5, color="gray", lw=1, linestyle=":", alpha=0.7)
        ax.set_title(province.split(" ", 1)[1][:11], fontsize=8)
        if pi == 0:
            ax.set_ylabel(zname, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", rotation=45, labelsize=7)

axes[0, 0].legend(fontsize=7)
plt.suptitle(
    "SINDy Rollout  |  blue=actual  red=predicted  dotted=test boundary",
    fontsize=10
)
plt.tight_layout()
plt.savefig(os.path.join(HERE, f"sindy_eval_trajectories_latent{LATENT_DIM}.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── Plot 3: Test MAE heatmap per province ────────────────────────
prov_mae = test_df.groupby("province")[[f"err_{z}" for z in z_cols]].mean()
prov_mae.columns = z_cols
fig, ax = plt.subplots(figsize=(6, max(6, len(prov_mae) * 0.3)))
im = ax.imshow(prov_mae.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(LATENT_DIM))
ax.set_xticklabels(z_cols)
ax.set_yticks(range(len(prov_mae)))
ax.set_yticklabels([p.split(" ", 1)[1][:20] for p in prov_mae.index], fontsize=7)
plt.colorbar(im, ax=ax, label="MAE")
ax.set_title("Test MAE per Province and Latent Axis")
plt.tight_layout()
plt.savefig(os.path.join(HERE, f"sindy_eval_test_heatmap_latent{LATENT_DIM}.png"),
            dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved sindy_eval_residuals_latent{LATENT_DIM}.png")
print(f"Saved sindy_eval_trajectories_latent{LATENT_DIM}.png")
print(f"Saved sindy_eval_test_heatmap_latent{LATENT_DIM}.png")
print("\n── DONE ─────────────────────────────────────────────────")
