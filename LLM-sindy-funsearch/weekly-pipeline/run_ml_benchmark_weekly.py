"""
run_ml_benchmark_weekly.py
────────────────────────────────────────────────────────────────
ML benchmark on weekly latent representations.

Objective: one-step-ahead weekly incidence prediction.
           Given latent z at week t, predict incidence at week t+1.

Feature set per sample:
  z1..zK          latent dims from weekly autoencoder
  time_sin        sin(2*pi*week/52)
  time_cos        cos(2*pi*week/52)
  province_id     integer label-encoded province
  (raw features excluded to keep the test clean)

Train:  province-weeks from years 2015-2019
Val:    province-weeks from years 2020-2021
Test:   province-weeks from years 2022-2023  (LOCKED until final eval)

Models tested:
  Linear family  : LinearRegression, Ridge, Lasso, ElasticNet,
                   BayesianRidge, HuberRegressor
  Kernel/KNN     : SVR (rbf), KNeighborsRegressor
  Tree ensembles : RandomForestRegressor, ExtraTreesRegressor,
                   GradientBoostingRegressor, AdaBoostRegressor,
                   HistGradientBoostingRegressor
  Neural         : MLPRegressor (2 hidden layers)
  Optional       : XGBoostRegressor, LightGBMRegressor (if installed)

Outputs (all in weekly-pipeline/):
  ml_benchmark_weekly_results.csv   per-model val and test R2
  ml_benchmark_weekly_byregion.csv  per-model per-province test R2
  ml_benchmark_weekly_plot.png      R2 bar chart (val + test)
  ml_benchmark_weekly_scatter/      scatter plots for top-3 models
────────────────────────────────────────────────────────────────
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, HuberRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, "extended_input_normalized.csv")

LATENT_DIR  = os.path.join(HERE, "latents")
SCATTER_DIR = os.path.join(HERE, "ml_benchmark_weekly_scatter")
os.makedirs(SCATTER_DIR, exist_ok=True)

LATENT_DIM  = 6       # change to 9 to use best-quality dim9 model
LATENT_CSV  = os.path.join(LATENT_DIR, f"latent_weekly_dim{LATENT_DIM}.csv")

TRAIN_YEARS = list(range(2015, 2020))
VAL_YEARS   = [2020, 2021]
TEST_YEARS  = [2022, 2023]

TOP_N_SCATTER = 3     # scatter plots for top-N models by val R2

print(f"Loading data (latent_dim={LATENT_DIM}) ...")

# ── Load raw weekly data for the target column ─────────────────
raw = pd.read_csv(DATA_CSV)
raw = raw.sort_values(["province", "year", "week"]).reset_index(drop=True)

# Target: next-week incidence.  Within each province shift by -1.
raw["target_next_week"] = (
    raw.groupby("province")["incidence"].shift(-1)
)

# ── Load latent representations (train+val only at this point) ─
latent = pd.read_csv(LATENT_CSV)
z_cols = [f"z{i+1}" for i in range(LATENT_DIM)]

# ── For test years we need to encode through the saved model ───
# The latent CSV only contains train+val rows.
# We encode test rows now using the saved model.
import torch
import torch.nn as nn

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

# Load model and encode test rows
MODEL_PATH = os.path.join(HERE, "models", f"best_model_weekly_dim{LATENT_DIM}.pt")
ID_COLS      = ["province", "year", "month", "week"]
FEATURE_COLS = [c for c in raw.columns if c not in ID_COLS
                and c != "target_next_week"]

# Circular time encoding
raw["time_sin"] = np.sin(2.0 * np.pi * raw["week"] / 52.0)
raw["time_cos"] = np.cos(2.0 * np.pi * raw["week"] / 52.0)
INPUT_FEATURES = FEATURE_COLS + ["time_sin", "time_cos"]
INPUT_DIM      = len(INPUT_FEATURES)

# NaN filling with train means
train_mask = raw["year"].isin(TRAIN_YEARS)
col_means  = raw.loc[train_mask, INPUT_FEATURES].mean()
raw[INPUT_FEATURES] = raw[INPUT_FEATURES].fillna(col_means)

HIDDEN_DIMS    = [512, 256]
DROPOUT        = 0.158
ACTIVATION     = "leakyrelu"
USE_BATCHNORM  = True

ae_model = WeeklyAutoencoder(
    INPUT_DIM, LATENT_DIM, HIDDEN_DIMS,
    DROPOUT, ACTIVATION, USE_BATCHNORM
)
ae_model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)
ae_model.eval()

test_mask = raw["year"].isin(TEST_YEARS)
X_test_raw = torch.tensor(
    raw.loc[test_mask, INPUT_FEATURES].values.astype(np.float32)
)
with torch.no_grad():
    Z_test = ae_model.encode(X_test_raw).numpy()

test_latent = pd.DataFrame(Z_test, columns=z_cols)
test_latent["province"] = raw.loc[test_mask, "province"].values
test_latent["year"]     = raw.loc[test_mask, "year"].values
test_latent["week"]     = raw.loc[test_mask, "week"].values
test_latent["split"]    = "test"

# Combine latent representations across all splits
all_latent = pd.concat([latent, test_latent], ignore_index=True)

# Merge target and time features from raw
merge_keys = ["province", "year", "week"]
raw_target = raw[merge_keys + ["target_next_week", "time_sin", "time_cos"]].copy()
all_latent = all_latent.merge(raw_target, on=merge_keys, how="left")

# Province label encoding
le = LabelEncoder()
all_latent["province_id"] = le.fit_transform(all_latent["province"])

# Drop rows where next-week target is missing (last week of each province)
all_latent = all_latent.dropna(subset=["target_next_week"]).reset_index(drop=True)

# ── Build feature matrix ───────────────────────────────────────
FEAT_COLS = z_cols + ["time_sin", "time_cos", "province_id"]
X_all = all_latent[FEAT_COLS].values.astype(np.float32)
y_all = all_latent["target_next_week"].values.astype(np.float32)
splits = all_latent["split"].values
provinces = all_latent["province"].values

train_idx = np.where(np.isin(splits, ["train"]))[0]
val_idx   = np.where(splits == "val")[0]
test_idx  = np.where(splits == "test")[0]

X_tr, y_tr = X_all[train_idx], y_all[train_idx]
X_va, y_va = X_all[val_idx],   y_all[val_idx]
X_te, y_te = X_all[test_idx],  y_all[test_idx]
prov_te    = provinces[test_idx]

print(f"  Train rows: {len(X_tr)}  Val rows: {len(X_va)}  Test rows: {len(X_te)}")
print(f"  Feature dim: {X_tr.shape[1]}  (z1..z{LATENT_DIM} + time_sin + time_cos + province_id)")
print()

# ── Model catalogue ────────────────────────────────────────────
MODELS_RAW = {
    "LinearRegression"            : LinearRegression(),
    "Ridge(alpha=1)"              : Ridge(alpha=1.0),
    "Ridge(alpha=10)"             : Ridge(alpha=10.0),
    "Lasso(alpha=0.001)"          : Lasso(alpha=0.001, max_iter=5000),
    "ElasticNet(l1=0.5)"          : ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
    "BayesianRidge"               : BayesianRidge(),
    "HuberRegressor"              : HuberRegressor(max_iter=300),
    "SVR(rbf,C=10)"               : SVR(kernel="rbf", C=10.0, epsilon=0.01),
    "SVR(rbf,C=100)"              : SVR(kernel="rbf", C=100.0, epsilon=0.01),
    "KNN(k=5)"                    : KNeighborsRegressor(n_neighbors=5),
    "KNN(k=15)"                   : KNeighborsRegressor(n_neighbors=15),
    "RandomForest(200)"           : RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "ExtraTrees(200)"             : ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting(200)"       : GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                               max_depth=4, random_state=42),
    "HistGradientBoosting"        : HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                                   max_depth=4, random_state=42),
    "AdaBoost(100)"               : AdaBoostRegressor(n_estimators=100, random_state=42),
    "MLP(256,128)"                : MLPRegressor(hidden_layer_sizes=(256, 128),
                                                  activation="relu", max_iter=500,
                                                  learning_rate_init=1e-3, random_state=42),
    "MLP(512,256,128)"            : MLPRegressor(hidden_layer_sizes=(512, 256, 128),
                                                  activation="relu", max_iter=500,
                                                  learning_rate_init=5e-4, random_state=42),
}

# Models that need StandardScaler applied first
NEEDS_SCALE = {
    "LinearRegression", "Ridge(alpha=1)", "Ridge(alpha=10)",
    "Lasso(alpha=0.001)", "ElasticNet(l1=0.5)", "BayesianRidge",
    "HuberRegressor", "SVR(rbf,C=10)", "SVR(rbf,C=100)",
    "MLP(256,128)", "MLP(512,256,128)",
}

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    MODELS_RAW["XGBoost"] = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0
    )
    print("  XGBoost found and added.")
except ImportError:
    print("  XGBoost not installed, skipping.")

# Optional LightGBM
try:
    from lightgbm import LGBMRegressor
    MODELS_RAW["LightGBM"] = LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=-1
    )
    print("  LightGBM found and added.")
except ImportError:
    print("  LightGBM not installed, skipping.")

print(f"\n  Running {len(MODELS_RAW)} models ...\n")

# ── Training and evaluation ────────────────────────────────────
results = []
preds_val  = {}
preds_test = {}

scaler = StandardScaler()
scaler.fit(X_tr)

X_tr_sc = scaler.transform(X_tr)
X_va_sc = scaler.transform(X_va)
X_te_sc = scaler.transform(X_te)

for name, model in MODELS_RAW.items():
    try:
        if name in NEEDS_SCALE:
            model.fit(X_tr_sc, y_tr)
            yhat_val  = model.predict(X_va_sc)
            yhat_test = model.predict(X_te_sc)
        else:
            model.fit(X_tr, y_tr)
            yhat_val  = model.predict(X_va)
            yhat_test = model.predict(X_te)

        r2_val  = float(r2_score(y_va, yhat_val))
        r2_test = float(r2_score(y_te, yhat_test))

        results.append({
            "model"  : name,
            "r2_val" : round(r2_val,  4),
            "r2_test": round(r2_test, 4),
        })
        preds_val[name]  = yhat_val
        preds_test[name] = yhat_test

        print(f"  {name:<32s}  val_r2={r2_val:+.4f}  test_r2={r2_test:+.4f}")

    except Exception as exc:
        print(f"  {name:<32s}  ERROR: {exc}")
        results.append({"model": name, "r2_val": float("nan"), "r2_test": float("nan")})

# ── Results table ──────────────────────────────────────────────
res_df = pd.DataFrame(results).sort_values("r2_val", ascending=False).reset_index(drop=True)
print("\n" + "=" * 70)
print("  BENCHMARK SUMMARY  (sorted by val R2)")
print("=" * 70)
print(res_df.to_string(index=False))

out_csv = os.path.join(HERE, "ml_benchmark_weekly_results.csv")
res_df.to_csv(out_csv, index=False)
print(f"\n  Results saved to ml_benchmark_weekly_results.csv")

# ── Per-province test R2 for all models ────────────────────────
prov_list  = sorted(set(prov_te))
by_region  = []
for name in preds_test:
    yh = preds_test[name]
    for prov in prov_list:
        mask = prov_te == prov
        if mask.sum() < 2:
            continue
        r2p = float(r2_score(y_te[mask], yh[mask]))
        by_region.append({
            "model"   : name,
            "province": prov,
            "r2_test" : round(r2p, 4),
        })

region_df = pd.DataFrame(by_region)
region_csv = os.path.join(HERE, "ml_benchmark_weekly_byregion.csv")
region_df.to_csv(region_csv, index=False)
print(f"  Per-province results saved to ml_benchmark_weekly_byregion.csv")

# Print per-province for top-5 models
top5 = res_df.head(5)["model"].tolist()
print("\n  Per-province test R2 for top-5 models by val_r2:")
pivot = region_df[region_df["model"].isin(top5)].pivot(
    index="province", columns="model", values="r2_test")
print(pivot.to_string())

# ── Bar chart: val and test R2 ─────────────────────────────────
plot_df = res_df.dropna(subset=["r2_val"]).copy()
n = len(plot_df)
x = np.arange(n)
width = 0.38

fig, ax = plt.subplots(figsize=(max(12, n * 0.7), 6))
bars1 = ax.bar(x - width / 2, plot_df["r2_val"],  width, label="Val R2",  color="#4C72B0", alpha=0.85)
bars2 = ax.bar(x + width / 2, plot_df["r2_test"], width, label="Test R2", color="#DD8452", alpha=0.85)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(plot_df["model"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("R2 score")
ax.set_title(f"Weekly ML Benchmark  latent_dim={LATENT_DIM}  one-step-ahead incidence")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(HERE, "ml_benchmark_weekly_plot.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\n  Bar chart saved to ml_benchmark_weekly_plot.png")

# ── Scatter plots for top-N models ────────────────────────────
top_names = res_df.dropna(subset=["r2_test"]).head(TOP_N_SCATTER)["model"].tolist()
for name in top_names:
    if name not in preds_test:
        continue
    yh = preds_test[name]
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

print(f"  Scatter plots saved to ml_benchmark_weekly_scatter/")

# ── Final summary ─────────────────────────────────────────────
best_val = res_df.iloc[0]
print("\n" + "=" * 70)
print("  BEST MODEL BY VAL R2")
print("=" * 70)
print(f"  Model    : {best_val['model']}")
print(f"  Val  R2  : {best_val['r2_val']:.4f}")
print(f"  Test R2  : {best_val['r2_test']:.4f}")
print()
print("  Done.  Test data was only unlocked at final evaluation above.")
print("─" * 70)
