"""
run_ml_benchmark_yearly.py
────────────────────────────────────────────────────────────────
ML benchmark on ANNUAL latent representations.

Objective: one-step-ahead ANNUAL incidence prediction.
           Given the latent z for (province, year t), predict
           mean weekly incidence for year t+1.

Feature set per sample:
  z1..z6          latent dims from the annual autoencoder (v1)
  province_id     integer label-encoded province
  year_norm       year normalised to [0,1] over 2015-2023

Train:  (province, year) pairs for years 2015-2019  (160 rows)
Val:    (province, year) pairs for years 2020-2021   (64 rows)
Test:   (province, year) pairs for years 2022-2023   (64 rows, LOCKED)

NOTE: sample counts are small.  Tree and neural models may overfit
to train and some variance is expected across runs.

Models:  same catalogue as run_ml_benchmark_weekly.py

Outputs (all in weekly-pipeline/):
  ml_benchmark_yearly_results.csv
  ml_benchmark_yearly_byregion.csv
  ml_benchmark_yearly_plot.png
  ml_benchmark_yearly_scatter/
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
AE_DIR   = os.path.join(ROOT, "autoencoder")
DATA_CSV = os.path.join(ROOT, "extended_input_normalized.csv")

LATENT_DIM    = 6
LATENT_CSV    = os.path.join(AE_DIR, f"latent_representations_latent{LATENT_DIM}.csv")
MODEL_PATH    = os.path.join(AE_DIR, f"best_model_latent{LATENT_DIM}.pt")
SCATTER_DIR   = os.path.join(HERE, "ml_benchmark_yearly_scatter")
os.makedirs(SCATTER_DIR, exist_ok=True)

WEEKS_PER_YEAR = 52
TRAIN_YEARS    = list(range(2015, 2020))
VAL_YEARS      = [2020, 2021]
TEST_YEARS     = [2022, 2023]
ALL_YEARS      = TRAIN_YEARS + VAL_YEARS + TEST_YEARS
TOP_N_SCATTER  = 3

print(f"Loading annual data (latent_dim={LATENT_DIM}) ...")

# ── Load raw data, build annual chunks for all years ──────────
raw = pd.read_csv(DATA_CSV)
ID_COLS      = ["province", "year", "month", "week"]
FEATURE_COLS = [c for c in raw.columns if c not in ID_COLS]
INPUT_DIM    = WEEKS_PER_YEAR * len(FEATURE_COLS)

raw = raw.sort_values(["province", "year", "week"]).reset_index(drop=True)

# Annual mean incidence per (province, year) for building targets
annual_inc = (
    raw.groupby(["province", "year"])["incidence"]
    .mean()
    .reset_index()
    .rename(columns={"incidence": "mean_inc"})
)

# Build flattened annual tensors for ALL years (need test for encoding)
chunks, chunk_labels = [], []
for (province, year), grp in raw.groupby(["province", "year"]):
    grp_s = grp.sort_values("week")
    if len(grp_s) < WEEKS_PER_YEAR:
        continue
    vec = grp_s[FEATURE_COLS].values[:WEEKS_PER_YEAR].flatten().astype(np.float32)
    chunks.append(vec)
    chunk_labels.append({"province": province, "year": year})

chunks = np.array(chunks, dtype=np.float32)
if np.isnan(chunks).sum() > 0:
    col_means = np.nanmean(chunks, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask  = np.isnan(chunks)
    chunks[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

labels_df = pd.DataFrame(chunk_labels)
print(f"  Total (province, year) chunks: {len(chunks)}")
print(f"  Input dim: {INPUT_DIM}  ({WEEKS_PER_YEAR} weeks x {len(FEATURE_COLS)} features)")

# ── Load annual autoencoder and encode test chunks ────────────
ACTS = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "elu": nn.ELU}

class DengueAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, pre_bottleneck,
                 dropout, activation, use_batchnorm, shallow_decoder=False):
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
                if dropout > 0.0:
                    enc_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*enc_layers)

        dec_dims = list(reversed(enc_dims))
        if shallow_decoder:
            dec_dims = [latent_dim, input_dim]
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

# v1 best config
ae = DengueAutoencoder(
    input_dim      = INPUT_DIM,
    latent_dim     = LATENT_DIM,
    hidden_dims    = [256, 512],
    pre_bottleneck = 32,
    dropout        = 0.0629,
    activation     = "elu",
    use_batchnorm  = True,
    shallow_decoder= False,
)
ae.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
ae.eval()
print(f"  Loaded annual autoencoder from {MODEL_PATH}")

# Encode all chunks (including test) to get z
with torch.no_grad():
    Z_all = ae.encode(torch.tensor(chunks)).numpy()

z_cols = [f"z{i+1}" for i in range(LATENT_DIM)]
all_latent = labels_df.copy()
for j, col in enumerate(z_cols):
    all_latent[col] = Z_all[:, j]

# ── Load saved train+val latents to verify alignment ──────────
saved = pd.read_csv(LATENT_CSV)
trainval_mask = all_latent["year"].isin(TRAIN_YEARS + VAL_YEARS)
print(f"  Train+val chunks re-encoded: {trainval_mask.sum()}  (saved had {len(saved)})")

# ── Assign split labels ────────────────────────────────────────
def year_to_split(y):
    if y in TRAIN_YEARS:
        return "train"
    if y in VAL_YEARS:
        return "val"
    return "test"

all_latent["split"] = all_latent["year"].apply(year_to_split)

# ── Build targets: next-year mean incidence ────────────────────
# Shift annual_inc by one year per province
annual_inc_shifted = annual_inc.copy()
annual_inc_shifted["year"] = annual_inc_shifted["year"] - 1
annual_inc_shifted = annual_inc_shifted.rename(columns={"mean_inc": "target_next_year"})

all_latent = all_latent.merge(
    annual_inc_shifted, on=["province", "year"], how="left"
)

# Drop rows without a next-year target (i.e., last year per province)
all_latent = all_latent.dropna(subset=["target_next_year"]).reset_index(drop=True)
print(f"  Rows after target join (exclude last year per province): {len(all_latent)}")

# ── Encode province and normalise year ────────────────────────
le = LabelEncoder()
all_latent["province_id"] = le.fit_transform(all_latent["province"])

year_min = min(ALL_YEARS)
year_max = max(ALL_YEARS)
all_latent["year_norm"] = (all_latent["year"] - year_min) / (year_max - year_min)

FEAT_COLS = z_cols + ["province_id", "year_norm"]
X_all = all_latent[FEAT_COLS].values.astype(np.float32)
y_all = all_latent["target_next_year"].values.astype(np.float32)
splits    = all_latent["split"].values
provinces = all_latent["province"].values

train_idx = np.where(splits == "train")[0]
val_idx   = np.where(splits == "val")[0]
test_idx  = np.where(splits == "test")[0]

X_tr, y_tr = X_all[train_idx], y_all[train_idx]
X_va, y_va = X_all[val_idx],   y_all[val_idx]
X_te, y_te = X_all[test_idx],  y_all[test_idx]
prov_te    = provinces[test_idx]

print(f"\n  Train: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}")
print(f"  Feature dim: {X_tr.shape[1]}  (z1..z{LATENT_DIM} + province_id + year_norm)")
print()

# ── Model catalogue (same as weekly benchmark) ────────────────
MODELS_RAW = {
    "LinearRegression"       : LinearRegression(),
    "Ridge(alpha=1)"         : Ridge(alpha=1.0),
    "Ridge(alpha=10)"        : Ridge(alpha=10.0),
    "Lasso(alpha=0.001)"     : Lasso(alpha=0.001, max_iter=5000),
    "ElasticNet(l1=0.5)"     : ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
    "BayesianRidge"          : BayesianRidge(),
    "HuberRegressor"         : HuberRegressor(max_iter=300),
    "SVR(rbf,C=10)"          : SVR(kernel="rbf", C=10.0, epsilon=0.01),
    "SVR(rbf,C=100)"         : SVR(kernel="rbf", C=100.0, epsilon=0.01),
    "KNN(k=3)"               : KNeighborsRegressor(n_neighbors=3),
    "KNN(k=5)"               : KNeighborsRegressor(n_neighbors=5),
    "RandomForest(200)"      : RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "ExtraTrees(200)"        : ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting(200)"  : GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                          max_depth=3, random_state=42),
    "HistGradientBoosting"   : HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                              max_depth=3, random_state=42),
    "AdaBoost(100)"          : AdaBoostRegressor(n_estimators=100, random_state=42),
    "MLP(128,64)"            : MLPRegressor(hidden_layer_sizes=(128, 64),
                                             activation="relu", max_iter=1000,
                                             learning_rate_init=1e-3, random_state=42),
    "MLP(256,128)"           : MLPRegressor(hidden_layer_sizes=(256, 128),
                                             activation="relu", max_iter=1000,
                                             learning_rate_init=5e-4, random_state=42),
}

NEEDS_SCALE = {
    "LinearRegression", "Ridge(alpha=1)", "Ridge(alpha=10)",
    "Lasso(alpha=0.001)", "ElasticNet(l1=0.5)", "BayesianRidge",
    "HuberRegressor", "SVR(rbf,C=10)", "SVR(rbf,C=100)",
    "MLP(128,64)", "MLP(256,128)",
}

try:
    from xgboost import XGBRegressor
    MODELS_RAW["XGBoost"] = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0
    )
    print("  XGBoost found and added.")
except ImportError:
    print("  XGBoost not installed, skipping.")

try:
    from lightgbm import LGBMRegressor
    MODELS_RAW["LightGBM"] = LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=-1
    )
    print("  LightGBM found and added.")
except ImportError:
    print("  LightGBM not installed, skipping.")

print(f"\n  Running {len(MODELS_RAW)} models ...\n")

# ── Scale ─────────────────────────────────────────────────────
scaler = StandardScaler()
scaler.fit(X_tr)
X_tr_sc = scaler.transform(X_tr)
X_va_sc = scaler.transform(X_va)
X_te_sc = scaler.transform(X_te)

# ── Train and evaluate ─────────────────────────────────────────
results   = []
preds_val  = {}
preds_test = {}

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
print("  BENCHMARK SUMMARY  yearly  (sorted by val R2)")
print("=" * 70)
print(res_df.to_string(index=False))

out_csv = os.path.join(HERE, "ml_benchmark_yearly_results.csv")
res_df.to_csv(out_csv, index=False)
print(f"\n  Results saved to ml_benchmark_yearly_results.csv")

# ── Per-province test R2 ───────────────────────────────────────
prov_list = sorted(set(prov_te))
by_region = []
for name in preds_test:
    yh = preds_test[name]
    for prov in prov_list:
        mask = prov_te == prov
        if mask.sum() < 1:
            continue
        r2p = float(r2_score(y_te[mask], yh[mask])) if mask.sum() >= 2 else float("nan")
        by_region.append({"model": name, "province": prov, "r2_test": round(r2p, 4) if not np.isnan(r2p) else float("nan")})

region_df = pd.DataFrame(by_region) if by_region else pd.DataFrame(columns=["model", "province", "r2_test"])
region_csv = os.path.join(HERE, "ml_benchmark_yearly_byregion.csv")
region_df.to_csv(region_csv, index=False)
print(f"  Per-province results saved to ml_benchmark_yearly_byregion.csv")

top5 = res_df.head(5)["model"].tolist()
print("\n  Per-province test R2 for top-5 models:")
if not region_df.empty and "model" in region_df.columns:
    pivot = region_df[region_df["model"].isin(top5)].pivot(
        index="province", columns="model", values="r2_test")
    print(pivot.to_string())
else:
    print("  Only one test year per province after target shift. Per-province R2 not available.")

# ── Bar chart ──────────────────────────────────────────────────
plot_df = res_df.dropna(subset=["r2_val"]).copy()
n = len(plot_df)
x = np.arange(n)
width = 0.38

fig, ax = plt.subplots(figsize=(max(12, n * 0.7), 6))
ax.bar(x - width / 2, plot_df["r2_val"],  width, label="Val R2",  color="#4C72B0", alpha=0.85)
ax.bar(x + width / 2, plot_df["r2_test"], width, label="Test R2", color="#DD8452", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(plot_df["model"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("R2 score")
ax.set_title(f"Annual ML Benchmark  latent_dim={LATENT_DIM}  next-year incidence")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(HERE, "ml_benchmark_yearly_plot.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\n  Bar chart saved to ml_benchmark_yearly_plot.png")

# ── Scatter plots for top-N models ────────────────────────────
top_names = res_df.dropna(subset=["r2_test"]).head(TOP_N_SCATTER)["model"].tolist()
for name in top_names:
    if name not in preds_test:
        continue
    yh  = preds_test[name]
    r2t = float(r2_score(y_te, yh))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_te, yh, s=30, alpha=0.6, color="#4C72B0")
    lo = min(y_te.min(), yh.min())
    hi = max(y_te.max(), yh.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("True next-year incidence")
    ax.set_ylabel("Predicted next-year incidence")
    ax.set_title(f"{name}  test_r2={r2t:.4f}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    safe = name.replace("(", "").replace(")", "").replace(",", "").replace(" ", "_")
    plt.savefig(os.path.join(SCATTER_DIR, f"scatter_{safe}.png"), dpi=150)
    plt.close()

print(f"  Scatter plots saved to ml_benchmark_yearly_scatter/")

# ── Final summary ──────────────────────────────────────────────
best_val = res_df.iloc[0]
print("\n" + "=" * 70)
print("  BEST MODEL BY VAL R2  (annual)")
print("=" * 70)
print(f"  Model    : {best_val['model']}")
print(f"  Val  R2  : {best_val['r2_val']:.4f}")
print(f"  Test R2  : {best_val['r2_test']:.4f}")
print()
print("  Done.  Test data was only unlocked at final evaluation above.")
print("─" * 70)
