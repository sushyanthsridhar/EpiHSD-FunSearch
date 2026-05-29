"""
run_gru_weekly.py
────────────────────────────────────────────────────────────────
Train and evaluate GRU models for weekly incidence prediction.

Run AFTER tune_gru_weekly.py. Copy best config values into the
CONFIG block below before running.

Two modes:
  1. GLOBAL GRU  trained on all 32 provinces together.
  2. CLUSTER GRU  one model per cluster (5 models).
     Province embedding is re-learned per cluster.

Train: 2015-2019   Val: 2020-2021   Test: 2022-2023 (LOCKED).

Run:
  python3 weekly-pipeline/run_gru_weekly.py

Outputs (weekly-pipeline/):
  gru_global_model.pt
  gru_cluster{k}_model.pt              for k in 0..4
  gru_results.csv                      val and test R2 summary
  gru_byregion.csv                     per-province test R2
  gru_training_loss_global.png
  gru_training_loss_cluster{k}.png
────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, "extended_input_normalized.csv")

LATENT_DIM   = 6
LATENT_CSV   = os.path.join(HERE, "latents", f"latent_weekly_dim{LATENT_DIM}.csv")
MODEL_PATH   = os.path.join(HERE, "models", f"best_model_weekly_dim{LATENT_DIM}.pt")
CLUSTER_CSV  = os.path.join(HERE, "province_clusters.csv")

TRAIN_YEARS = list(range(2015, 2020))
VAL_YEARS   = [2020, 2021]
TEST_YEARS  = [2022, 2023]

# ── CONFIG: best config from trial 42 of tune_gru_weekly.py ───
SEQ_LEN      = 8
HIDDEN_SIZE  = 256
NUM_LAYERS   = 1
DROPOUT      = 0.261
LR           = 0.00035
BATCH_SIZE   = 64
WEIGHT_DECAY = 0.000152
PROV_EMB_DIM = 4
EPOCHS       = 400    # increased, GRU needs more epochs than RF
PATIENCE     = 40     # increased to give model room to converge
# ─────────────────────────────────────────────────────────────

# ── Encode test rows ──────────────────────────────────────────
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
    def forward(self, x): return self.decoder(self.encoder(x))
    def encode(self, x):  return self.encoder(x)

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

raw = pd.read_csv(DATA_CSV)
raw = raw.sort_values(["province", "year", "week"]).reset_index(drop=True)
ID_COLS      = ["province", "year", "month", "week"]
FEATURE_COLS = [c for c in raw.columns if c not in ID_COLS]
raw["time_sin"] = np.sin(2.0 * np.pi * raw["week"] / 52.0)
raw["time_cos"] = np.cos(2.0 * np.pi * raw["week"] / 52.0)
INPUT_FEATURES = FEATURE_COLS + ["time_sin", "time_cos"]
INPUT_DIM = len(INPUT_FEATURES)
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

z_cols = [f"z{i+1}" for i in range(LATENT_DIM)]
test_latent = pd.DataFrame(Z_test, columns=z_cols)
test_latent["province"] = raw.loc[test_mask, "province"].values
test_latent["year"]     = raw.loc[test_mask, "year"].values
test_latent["week"]     = raw.loc[test_mask, "week"].values
test_latent["split"]    = "test"

saved  = pd.read_csv(LATENT_CSV)
latent = pd.concat([saved, test_latent], ignore_index=True)
latent = latent.merge(
    raw[["province", "year", "week", "incidence", "time_sin", "time_cos"]],
    on=["province", "year", "week"], how="left"
)
latent = latent.sort_values(["province", "year", "week"]).reset_index(drop=True)

clusters   = pd.read_csv(CLUSTER_CSV)[["province", "cluster_id"]]
latent     = latent.merge(clusters, on="province", how="left")
provinces  = sorted(latent["province"].unique())
prov2idx   = {p: i for i, p in enumerate(provinces)}
N_PROV_GLOBAL = len(provinces)
latent["prov_idx"] = latent["province"].map(prov2idx)

FEAT_COLS = z_cols + ["incidence", "time_sin", "time_cos"]
N_FEATS   = len(FEAT_COLS)
print(f"Global provinces: {N_PROV_GLOBAL}  Latent dim: {LATENT_DIM}")
print(f"Features per timestep: {N_FEATS}  Seq len: {SEQ_LEN}\n")

# ── Sequence builder ──────────────────────────────────────────
def build_sequences(df, year_list, seq_len):
    seqs, prov_ids, targets = [], [], []
    sel = df[df["year"].isin(year_list)]
    for prov, grp in sel.groupby("province"):
        grp = grp.sort_values(["year", "week"]).reset_index(drop=True)
        X   = grp[FEAT_COLS].values.astype(np.float32)
        y   = grp["incidence"].values.astype(np.float32)
        pid = int(grp["prov_idx"].iloc[0])
        for t in range(seq_len, len(X)):
            seqs.append(X[t - seq_len: t])
            prov_ids.append(pid)
            targets.append(y[t])
    if not seqs:
        return None, None, None
    return (np.array(seqs, dtype=np.float32),
            np.array(prov_ids, dtype=np.int64),
            np.array(targets,  dtype=np.float32))

# ── GRU model ─────────────────────────────────────────────────
class DengueGRU(nn.Module):
    def __init__(self, n_feats, n_provinces, prov_emb_dim,
                 hidden_size, num_layers, dropout):
        super().__init__()
        self.prov_emb  = nn.Embedding(n_provinces, prov_emb_dim)
        input_size     = n_feats + prov_emb_dim
        self.gru       = nn.GRU(input_size, hidden_size, num_layers,
                                batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0)
        self.drop      = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size, 1)

    def forward(self, x_seq, prov_ids):
        emb = self.prov_emb(prov_ids)
        emb = emb.unsqueeze(1).expand(-1, x_seq.size(1), -1)
        x   = torch.cat([x_seq, emb], dim=-1)
        out, _ = self.gru(x)
        return self.fc(self.drop(out[:, -1, :])).squeeze(-1)

# ── Training function ─────────────────────────────────────────
def train_gru(latent_df, year_train, year_val, n_provinces,
              model_save_path, label=""):
    X_tr, p_tr, y_tr = build_sequences(latent_df, year_train, SEQ_LEN)
    X_va, p_va, y_va = build_sequences(latent_df, year_val,   SEQ_LEN)
    if X_tr is None:
        print(f"  {label}: no training data, skipping.")
        return None, [], []

    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(p_tr), torch.tensor(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True)

    model  = DengueGRU(N_FEATS, n_provinces, PROV_EMB_DIM,
                       HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    loss_fn = nn.MSELoss()

    X_va_t = torch.tensor(X_va)
    p_va_t = torch.tensor(p_va)
    y_va_t = torch.tensor(y_va)

    train_losses, val_losses = [], []
    best_val, best_state, patience_c = float("inf"), None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_l = []
        for xb, pb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb, pb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_l.append(loss.item())
        train_loss = float(np.mean(batch_l))

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t, p_va_t), y_va_t).item()

        sched.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1

        if epoch % 25 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                yp = model(X_va_t, p_va_t).numpy()
            r2v = float(r2_score(y_va, yp))
            print(f"  {label}  Epoch {epoch:4d}  train={train_loss:.5f}"
                  f"  val={val_loss:.5f}  val_r2={r2v:.4f}"
                  + ("  *" if patience_c == 0 else ""))

        if patience_c >= PATIENCE:
            print(f"  {label}  Early stop at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, model_save_path)
    return model, train_losses, val_losses

def evaluate_gru(model, latent_df, year_list, label=""):
    X, p, y = build_sequences(latent_df, year_list, SEQ_LEN)
    if X is None:
        return float("nan"), {}, np.array([]), np.array([])
    model.eval()
    with torch.no_grad():
        yhat = model(torch.tensor(X), torch.tensor(p)).numpy()
    r2_global = float(r2_score(y, yhat))

    # Per-province R2
    prov_r2 = {}
    provs = latent_df[latent_df["year"].isin(year_list)]["province"].unique()
    for prov in sorted(provs):
        mask = latent_df[latent_df["year"].isin(year_list)].groupby(
            "province").groups.get(prov, [])
        # rebuild per province
        grp = latent_df[latent_df["province"] == prov]
        grp = grp[grp["year"].isin(year_list)].sort_values(["year", "week"])
        Xp  = grp[FEAT_COLS].values.astype(np.float32)
        yp  = grp["incidence"].values.astype(np.float32)
        pid = int(grp["prov_idx"].iloc[0])
        seqs_p, ids_p, tgts_p = [], [], []
        for t in range(SEQ_LEN, len(Xp)):
            seqs_p.append(Xp[t - SEQ_LEN: t])
            ids_p.append(pid)
            tgts_p.append(yp[t])
        if len(seqs_p) < 2:
            continue
        with torch.no_grad():
            yhp = model(
                torch.tensor(np.array(seqs_p, dtype=np.float32)),
                torch.tensor(np.array(ids_p,  dtype=np.int64))
            ).numpy()
        prov_r2[prov] = round(float(r2_score(np.array(tgts_p), yhp)), 4)

    print(f"  {label}  pooled_r2={r2_global:+.4f}")
    return r2_global, prov_r2, y, yhat

# ════════════════════════════════════════════════════════════════
# 1. GLOBAL GRU
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("  GLOBAL GRU  (all 32 provinces)")
print("=" * 60)

global_model, gl_tr_loss, gl_va_loss = train_gru(
    latent, TRAIN_YEARS, VAL_YEARS,
    n_provinces  = N_PROV_GLOBAL,
    model_save_path = os.path.join(HERE, "gru_global_model.pt"),
    label = "Global"
)

results = []
byregion = []

if global_model is not None:
    r2_val_g,  prov_r2_val_g,  _, _  = evaluate_gru(global_model, latent, VAL_YEARS,  "Global val")
    r2_test_g, prov_r2_test_g, _, _  = evaluate_gru(global_model, latent, TEST_YEARS, "Global test")

    results.append({"model": "GRU_global",
                    "r2_val": round(r2_val_g, 4),
                    "r2_test": round(r2_test_g, 4)})

    for prov, r2v in prov_r2_val_g.items():
        byregion.append({"model": "GRU_global", "province": prov,
                         "r2_val": r2v,
                         "r2_test": prov_r2_test_g.get(prov, float("nan"))})

    # Loss plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gl_tr_loss, label="Train")
    ax.plot(gl_va_loss, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"GRU Global  val_r2={r2_val_g:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "gru_training_loss_global.png"), dpi=150)
    plt.close()

# ════════════════════════════════════════════════════════════════
# 2. CLUSTER GRUs
# ════════════════════════════════════════════════════════════════
cluster_ids = sorted(latent["cluster_id"].dropna().unique())

for cid in cluster_ids:
    cid = int(cid)
    c_df    = latent[latent["cluster_id"] == cid].copy()
    c_provs = sorted(c_df["province"].unique())
    n_cprov = len(c_provs)

    # Re-index province IDs within cluster (0..n_cprov-1)
    cprov2idx = {p: i for i, p in enumerate(c_provs)}
    c_df = c_df.copy()
    c_df["prov_idx"] = c_df["province"].map(cprov2idx)

    print("\n" + "=" * 60)
    print(f"  CLUSTER {cid} GRU  ({n_cprov} provinces: {', '.join(c_provs)})")
    print("=" * 60)

    c_model, c_tr, c_va = train_gru(
        c_df, TRAIN_YEARS, VAL_YEARS,
        n_provinces     = n_cprov,
        model_save_path = os.path.join(HERE, f"gru_cluster{cid}_model.pt"),
        label           = f"Cluster{cid}"
    )
    if c_model is None:
        continue

    r2_val_c,  pv_r2_val_c,  _, _ = evaluate_gru(c_model, c_df, VAL_YEARS,  f"Cluster{cid} val")
    r2_test_c, pv_r2_test_c, _, _ = evaluate_gru(c_model, c_df, TEST_YEARS, f"Cluster{cid} test")

    results.append({"model": f"GRU_cluster{cid}",
                    "r2_val": round(r2_val_c, 4),
                    "r2_test": round(r2_test_c, 4)})

    for prov, r2v in pv_r2_val_c.items():
        byregion.append({"model": f"GRU_cluster{cid}", "province": prov,
                         "r2_val": r2v,
                         "r2_test": pv_r2_test_c.get(prov, float("nan"))})

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(c_tr, label="Train")
    ax.plot(c_va, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"GRU Cluster{cid}  ({n_cprov} prov)  val_r2={r2_val_c:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, f"gru_training_loss_cluster{cid}.png"), dpi=150)
    plt.close()

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  GRU RESULTS SUMMARY")
print("=" * 60)
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
res_df.to_csv(os.path.join(HERE, "gru_results.csv"), index=False)

region_df = pd.DataFrame(byregion)
region_df.to_csv(os.path.join(HERE, "gru_byregion.csv"), index=False)
print(f"\n  Saved gru_results.csv  and  gru_byregion.csv")
print("─" * 60)
