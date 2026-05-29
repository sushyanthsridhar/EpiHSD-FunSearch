"""
run_ml_benchmark_weekly_v2.py
────────────────────────────────────────────────────────────────
Improved ML benchmark on weekly latent representations.

Key improvements over v1:
  1. Lagged incidence features: incidence at t, t-1, t-2, t-3, t-4.
     Dengue is strongly autocorrelated. Without these the latent
     alone has no anchor for next-week prediction.
  2. Per-province model training for tree-based methods.
     Province heterogeneity is too large to pool globally.
  3. Persistence baseline: predict next week = this week.
     Everything must beat this to be meaningful.

Feature set per sample:
  z1..zK          latent dims from weekly autoencoder
  inc_lag0        incidence at week t   (current)
  inc_lag1..4     incidence at t-1, t-2, t-3, t-4
  time_sin        sin(2*pi*week/52)
  time_cos        cos(2*pi*week/52)
  province_id     integer label-encoded province

Train:  2015-2019   Val: 2020-2021   Test: 2022-2023 (LOCKED)

Outputs:
  ml_v2_global_results.csv        global model comparison
  ml_v2_perprovince_results.csv   per-province model results
  ml_v2_summary_plot.png          bar chart
  ml_v2_scatter/                  scatter plots for top models
────────────────────────────────────────────────────────────────
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score

from sklearn.linear_model import Ridge, Lasso, BayesianRidge, HuberRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor,
)
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, "extended_input_normalized.csv")
LATENT_DIR = os.path.join(HERE, "latents")
SCATTER_DIR = os.path.join(HERE, "ml_v2_scatter")
os.makedirs(SCATTER_DIR, exist_ok=True)

LATENT_DIM  = 6
LATENT_CSV  = os.path.join(LATENT_DIR, f"latent_weekly_dim{LATENT_DIM}.csv")
MODEL_PATH  = os.path.join(HERE, "models", f"best_model_weekly_dim{LATENT_DIM}.pt")

TRAIN_YEARS = list(range(2015, 2020))
VAL_YEARS   = [2020, 2021]
TEST_YEARS  = [2022, 2023]
N_LAGS      = 4    # incidence lags to include: t, t-1 ... t-N_LAGS

print(f"Loading data (latent_dim={LATENT_DIM}) ...")

# ── Load raw weekly data ───────────────────────────────────────
raw = pd.read_csv(DATA_CSV)
raw = raw.sort_values(["province", "year", "week"]).reset_index(drop=True)

ID_COLS      = ["province", "year", "month", "week"]
FEATURE_COLS = [c for c in raw.columns if c not in ID_COLS]

# Circular time encoding
raw["time_sin"] = np.sin(2.0 * np.pi * raw["week"] / 52.0)
raw["time_cos"] = np.cos(2.0 * np.pi * raw["week"] / 52.0)
INPUT_FEATURES = FEATURE_COLS + ["time_sin", "time_cos"]

# Fill NaNs with train column means
train_mask = raw["year"].isin(TRAIN_YEARS)
col_means  = raw.loc[train_mask, INPUT_FEATURES].mean()
raw[INPUT_FEATURES] = raw[INPUT_FEATURES].fillna(col_means)

# Next-week target
raw["target_next_week"] = raw.groupby("province")["incidence"].shift(-1)

# Lagged incidence features within each province
for lag in range(N_LAGS + 1):
    raw[f"inc_lag{lag}"] = raw.groupby("province")["incidence"].shift(lag)

LAG_COLS = [f"inc_lag{lag}" for lag in range(N_LAGS + 1)]

# ── Encode test rows through saved autoencoder ─────────────────
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

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

INPUT_DIM = len(INPUT_FEATURES)
ae = WeeklyAutoencoder(INPUT_DIM, LATENT_DIM, [512, 256], 0.158, "leakyrelu", True)
ae.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
ae.eval()

test_mask = raw["year"].isin(TEST_YEARS)
X_test_raw = torch.tensor(raw.loc[test_mask, INPUT_FEATURES].values.astype(np.float32))
with torch.no_grad():
    Z_test = ae.encode(X_test_raw).numpy()

z_cols = [f"z{i+1}" for i in range(LATENT_DIM)]
test_latent = pd.DataFrame(Z_test, columns=z_cols)
test_latent["province"] = raw.loc[test_mask, "province"].values
test_latent["year"]     = raw.loc[test_mask, "year"].values
test_latent["week"]     = raw.loc[test_mask, "week"].values
test_latent["split"]    = "test"

# Combine with saved train+val latents
saved_latent = pd.read_csv(LATENT_CSV)
all_latent = pd.concat([saved_latent, test_latent], ignore_index=True)

# Merge lag features and target from raw
merge_cols = ["province", "year", "week"]
extras = raw[merge_cols + ["target_next_week", "time_sin", "time_cos"] + LAG_COLS].copy()
all_latent = all_latent.merge(extras, on=merge_cols, how="left")

# Province encoding
le = LabelEncoder()
all_latent["province_id"] = le.fit_transform(all_latent["province"])

# Drop rows missing lags or target
drop_cols = ["target_next_week"] + LAG_COLS
all_latent = all_latent.dropna(subset=drop_cols).reset_index(drop=True)

FEAT_COLS = z_cols + LAG_COLS + ["time_sin", "time_cos", "province_id"]
X_all = all_latent[FEAT_COLS].values.astype(np.float32)
y_all = all_latent["target_next_week"].values.astype(np.float32)
splits    = all_latent["split"].values
provinces = all_latent["province"].values

train_idx = np.where(splits == "train")[0]
val_idx   = np.where(splits == "val")[0]
test_idx  = np.where(splits == "test")[0]

X_tr, y_tr = X_all[train_idx], y_all[train_idx]
X_va, y_va = X_all[val_idx],   y_all[val_idx]
X_te, y_te = X_all[test_idx],  y_all[test_idx]
prov_va    = provinces[val_idx]
prov_te    = provinces[test_idx]

print(f"  Train: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}")
print(f"  Features: {X_tr.shape[1]}  ({z_cols[0]}..z{LATENT_DIM}, inc_lag0..{N_LAGS}, time_sin, time_cos, province_id)")
print()

# ── Persistence baseline ───────────────────────────────────────
# Predict next week = this week (inc_lag0)
lag0_idx = FEAT_COLS.index("inc_lag0")
r2_persist_val  = r2_score(y_va, X_va[:, lag0_idx])
r2_persist_test = r2_score(y_te, X_te[:, lag0_idx])
print(f"  Persistence baseline  val_r2={r2_persist_val:+.4f}  test_r2={r2_persist_test:+.4f}")
print()

# ── Global model catalogue ────────────────────────────────────
MODELS = {
    "Persistence(baseline)"   : None,
    "Ridge(alpha=1)"          : Ridge(alpha=1.0),
    "BayesianRidge"           : BayesianRidge(),
    "Lasso(alpha=0.001)"      : Lasso(alpha=0.001, max_iter=5000),
    "HuberRegressor"          : HuberRegressor(max_iter=300),
    "RandomForest(200)"       : RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "ExtraTrees(200)"         : ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting(200)"   : GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                           max_depth=4, random_state=42),
    "HistGradientBoosting"    : HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                               max_depth=4, random_state=42),
    "MLP(256,128)"            : MLPRegressor(hidden_layer_sizes=(256, 128), activation="relu",
                                              max_iter=500, learning_rate_init=1e-3, random_state=42),
}

try:
    from xgboost import XGBRegressor
    MODELS["XGBoost"] = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                      subsample=0.8, colsample_bytree=0.8,
                                      random_state=42, n_jobs=-1, verbosity=0)
    print("  XGBoost added.")
except ImportError:
    pass

try:
    from lightgbm import LGBMRegressor
    MODELS["LightGBM"] = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                        subsample=0.8, colsample_bytree=0.8,
                                        random_state=42, n_jobs=-1, verbosity=-1)
    print("  LightGBM added.")
except ImportError:
    pass

NEEDS_SCALE = {"Ridge(alpha=1)", "BayesianRidge", "Lasso(alpha=0.001)",
               "HuberRegressor", "MLP(256,128)"}

scaler = StandardScaler()
scaler.fit(X_tr)
X_tr_sc = scaler.transform(X_tr)
X_va_sc = scaler.transform(X_va)
X_te_sc = scaler.transform(X_te)

print(f"  Running {len(MODELS)} global models ...\n")

global_results = []
preds_val_g  = {}
preds_test_g = {}

for name, model in MODELS.items():
    try:
        if model is None:
            yhat_val  = X_va[:, lag0_idx]
            yhat_test = X_te[:, lag0_idx]
        elif name in NEEDS_SCALE:
            model.fit(X_tr_sc, y_tr)
            yhat_val  = model.predict(X_va_sc)
            yhat_test = model.predict(X_te_sc)
        else:
            model.fit(X_tr, y_tr)
            yhat_val  = model.predict(X_va)
            yhat_test = model.predict(X_te)

        r2v = float(r2_score(y_va, yhat_val))
        r2t = float(r2_score(y_te, yhat_test))
        global_results.append({"model": name, "r2_val": round(r2v, 4), "r2_test": round(r2t, 4)})
        preds_val_g[name]  = yhat_val
        preds_test_g[name] = yhat_test
        print(f"  {name:<30s}  val_r2={r2v:+.4f}  test_r2={r2t:+.4f}")

    except Exception as exc:
        print(f"  {name:<30s}  ERROR: {exc}")
        global_results.append({"model": name, "r2_val": float("nan"), "r2_test": float("nan")})

global_df = pd.DataFrame(global_results).sort_values("r2_val", ascending=False).reset_index(drop=True)
print("\n" + "=" * 70)
print("  GLOBAL BENCHMARK  (sorted by val R2)")
print("=" * 70)
print(global_df.to_string(index=False))
global_df.to_csv(os.path.join(HERE, "ml_v2_global_results.csv"), index=False)

# ── Per-province models ────────────────────────────────────────
print("\n" + "=" * 70)
print("  PER-PROVINCE MODELS  (ExtraTrees, RandomForest, HistGB)")
print("=" * 70)

PP_MODELS = {
    "PP_ExtraTrees"       : lambda: ExtraTreesRegressor(n_estimators=200, random_state=42),
    "PP_RandomForest"     : lambda: RandomForestRegressor(n_estimators=200, random_state=42),
    "PP_HistGradientBoosting": lambda: HistGradientBoostingRegressor(max_iter=300,
                                                                      learning_rate=0.05,
                                                                      max_depth=4,
                                                                      random_state=42),
}

pp_results  = []
pp_preds_te = {n: np.zeros(len(y_te)) for n in PP_MODELS}

prov_list = sorted(set(provinces))
for prov in prov_list:
    tr_m = (provinces[train_idx] == prov)
    va_m = (provinces[val_idx]   == prov)
    te_m = (provinces[test_idx]  == prov)

    Xp_tr, yp_tr = X_tr[tr_m], y_tr[tr_m]
    Xp_va, yp_va = X_va[va_m], y_va[va_m]
    Xp_te, yp_te = X_te[te_m], y_te[te_m]

    if len(Xp_tr) < 10:
        continue

    for name, factory in PP_MODELS.items():
        mdl = factory()
        mdl.fit(Xp_tr, yp_tr)
        yh_va = mdl.predict(Xp_va)
        yh_te = mdl.predict(Xp_te)
        pp_preds_te[name][te_m] = yh_te

        r2v = float(r2_score(yp_va, yh_va)) if len(yp_va) >= 2 else float("nan")
        r2t = float(r2_score(yp_te, yh_te)) if len(yp_te) >= 2 else float("nan")
        pp_results.append({"model": name, "province": prov,
                            "r2_val": round(r2v, 4), "r2_test": round(r2t, 4)})

pp_df = pd.DataFrame(pp_results)

# Summary by model
pp_summary = pp_df.groupby("model")[["r2_val", "r2_test"]].mean().round(4)
print("\n  Mean across provinces:")
print(pp_summary.to_string())

# Overall test R2 using pooled predictions
for name in PP_MODELS:
    r2_pool = float(r2_score(y_te, pp_preds_te[name]))
    print(f"  {name}  pooled test_r2={r2_pool:+.4f}")
    global_results.append({"model": name,
                            "r2_val": float(pp_df[pp_df["model"] == name]["r2_val"].mean()),
                            "r2_test": r2_pool})

pp_df.to_csv(os.path.join(HERE, "ml_v2_perprovince_results.csv"), index=False)
print(f"\n  Per-province results saved to ml_v2_perprovince_results.csv")

# ── Final combined summary ─────────────────────────────────────
all_results_df = pd.DataFrame(global_results).sort_values("r2_val", ascending=False).reset_index(drop=True)
print("\n" + "=" * 70)
print("  COMBINED SUMMARY  (global + per-province models, sorted by val R2)")
print("=" * 70)
print(all_results_df.to_string(index=False))

# ── Bar chart ──────────────────────────────────────────────────
plot_df = all_results_df.dropna(subset=["r2_val"]).copy()
n = len(plot_df)
x = np.arange(n)
w = 0.38

fig, ax = plt.subplots(figsize=(max(14, n * 0.8), 6))
ax.bar(x - w/2, plot_df["r2_val"],  w, label="Val R2",  color="#4C72B0", alpha=0.85)
ax.bar(x + w/2, plot_df["r2_test"], w, label="Test R2", color="#DD8452", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(plot_df["model"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("R2 score")
ax.set_title(f"Weekly ML Benchmark v2  latent_dim={LATENT_DIM}  with lag features")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "ml_v2_summary_plot.png"), dpi=150)
plt.close()
print(f"\n  Plot saved to ml_v2_summary_plot.png")

# ── Scatter plots for top-3 global models ─────────────────────
top3 = global_df.dropna(subset=["r2_test"]).head(3)["model"].tolist()
for name in top3:
    if name not in preds_test_g:
        continue
    yh  = preds_test_g[name]
    r2t = float(r2_score(y_te, yh))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_te, yh, s=5, alpha=0.3, color="#4C72B0")
    lo = min(y_te.min(), yh.min())
    hi = max(y_te.max(), yh.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("True incidence (week t+1)")
    ax.set_ylabel("Predicted incidence (week t+1)")
    ax.set_title(f"{name}  test_r2={r2t:.4f}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe = name.replace("(", "").replace(")", "").replace(",", "").replace(" ", "_")
    plt.savefig(os.path.join(SCATTER_DIR, f"scatter_{safe}.png"), dpi=150)
    plt.close()
print(f"  Scatter plots saved to ml_v2_scatter/")

best = all_results_df.iloc[0]
print("\n" + "=" * 70)
print(f"  BEST: {best['model']}  val_r2={best['r2_val']:.4f}  test_r2={best['r2_test']:.4f}")
print("=" * 70)
