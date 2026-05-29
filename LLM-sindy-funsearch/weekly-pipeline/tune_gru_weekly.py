"""
tune_gru_weekly.py
────────────────────────────────────────────────────────────────
Bayesian hyperparameter tuning for the weekly GRU model.

Tuning is done on the GLOBAL dataset (all 32 provinces) so the
best config applies uniformly to both the global baseline model
and the per-cluster models in run_gru_weekly.py.

Task: one-step-ahead weekly incidence prediction.
Input sequence: last SEQ_LEN weeks of
  z1..z6  (weekly autoencoder latent)
  incidence at t  (current week, strong autocorrelation signal)
  time_sin, time_cos  (seasonal position)
  province_embedding  (learned, dim PROV_EMB_DIM)
Output: incidence at week t+1.

Train: 2015-2019   Val: 2020-2021   Test: locked.

Run:
  python3 weekly-pipeline/tune_gru_weekly.py

Outputs:
  weekly-pipeline/optuna_results_gru_weekly.csv
  weekly-pipeline/best_config_gru_weekly.txt
────────────────────────────────────────────────────────────────
"""

import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("Run:  pip install optuna  then retry.")

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, "extended_input_normalized.csv")

LATENT_DIM   = 6
LATENT_CSV   = os.path.join(HERE, "latents", f"latent_weekly_dim{LATENT_DIM}.csv")
MODEL_PATH   = os.path.join(HERE, "models", f"best_model_weekly_dim{LATENT_DIM}.pt")

TRAIN_YEARS  = list(range(2015, 2020))
VAL_YEARS    = [2020, 2021]
TEST_YEARS   = [2022, 2023]

N_TRIALS     = 60
MAX_EPOCHS   = 80
PATIENCE     = 15
PROV_EMB_DIM = 4     # province embedding size

# ── Encode test rows (needed only to fill the latent table) ───
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

saved = pd.read_csv(LATENT_CSV)
latent = pd.concat([saved, test_latent], ignore_index=True)

# Merge incidence target and time features
merge_cols = ["province", "year", "week"]
raw_extra  = raw[merge_cols + ["incidence", "time_sin", "time_cos"]].copy()
latent = latent.merge(raw_extra, on=merge_cols, how="left")
latent = latent.sort_values(["province", "year", "week"]).reset_index(drop=True)

# Province integer encoding
provinces   = sorted(latent["province"].unique())
prov2idx    = {p: i for i, p in enumerate(provinces)}
N_PROVINCES = len(provinces)
latent["prov_idx"] = latent["province"].map(prov2idx)

print(f"Provinces: {N_PROVINCES}  Latent dim: {LATENT_DIM}")
print(f"Total rows: {len(latent)}")

# ── Sliding window dataset builder ────────────────────────────
FEAT_COLS = z_cols + ["incidence", "time_sin", "time_cos"]
N_FEATS   = len(FEAT_COLS)   # features per timestep (excluding province emb)

def build_sequences(df, year_list, seq_len):
    """
    For each province, build (seq, prov_idx, target) triples.
    Sequence is strictly within one province, no year boundaries crossed.
    """
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
        self.prov_emb = nn.Embedding(n_provinces, prov_emb_dim)
        input_size    = n_feats + prov_emb_dim
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x_seq, prov_ids):
        # x_seq: (B, T, N_FEATS)   prov_ids: (B,)
        emb = self.prov_emb(prov_ids)              # (B, emb_dim)
        emb = emb.unsqueeze(1).expand(-1, x_seq.size(1), -1)
        x   = torch.cat([x_seq, emb], dim=-1)      # (B, T, N_FEATS+emb_dim)
        out, _ = self.gru(x)
        out     = self.dropout(out[:, -1, :])       # last timestep
        return self.fc(out).squeeze(-1)

# ── Optuna objective ───────────────────────────────────────────
def objective(trial):
    seq_len     = trial.suggest_categorical("seq_len",     [4, 8, 12, 16])
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
    num_layers  = trial.suggest_int("num_layers", 1, 3)
    dropout     = trial.suggest_float("dropout",  0.0, 0.4)
    lr          = trial.suggest_float("lr",       1e-4, 5e-3, log=True)
    batch_size  = trial.suggest_categorical("batch_size", [64, 128, 256])
    weight_decay= trial.suggest_float("weight_decay", 0.0, 1e-3)

    X_tr, p_tr, y_tr = build_sequences(latent, TRAIN_YEARS, seq_len)
    X_va, p_va, y_va = build_sequences(latent, VAL_YEARS,   seq_len)
    if X_tr is None or X_va is None:
        return float("nan")

    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(p_tr), torch.tensor(y_tr)),
        batch_size=batch_size, shuffle=True)

    model = DengueGRU(N_FEATS, N_PROVINCES, PROV_EMB_DIM,
                      hidden_size, num_layers, dropout)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    X_va_t = torch.tensor(X_va)
    p_va_t = torch.tensor(p_va)
    y_va_t = torch.tensor(y_va)

    best_val, patience_c = float("inf"), 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, pb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb, pb), yb)
            if torch.isnan(loss):
                return float("nan")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t, p_va_t), y_va_t).item()

        if val_loss < best_val:
            best_val   = val_loss
            patience_c = 0
        else:
            patience_c += 1
        if patience_c >= PATIENCE:
            break

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Compute val R2
    model.eval()
    with torch.no_grad():
        y_pred = model(X_va_t, p_va_t).numpy()
    ss_res = ((y_va - y_pred) ** 2).sum()
    ss_tot = ((y_va - y_va.mean()) ** 2).sum()
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8))

    trial.set_user_attr("val_r2",    round(r2, 4))
    trial.set_user_attr("epochs_run", epoch)
    return r2

# ── Run study ─────────────────────────────────────────────────
sampler = optuna.samplers.TPESampler(seed=42)
pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
study   = optuna.create_study(direction="maximize",
                               sampler=sampler, pruner=pruner)

trial_count = [0]
def callback(study, trial):
    trial_count[0] += 1
    if trial.state == optuna.trial.TrialState.COMPLETE:
        print(f"  Trial {trial_count[0]:3d}/{N_TRIALS}"
              f"  val_r2={trial.value:.4f}"
              f"  seq={trial.params.get('seq_len')}"
              f"  hidden={trial.params.get('hidden_size')}"
              f"  layers={trial.params.get('num_layers')}"
              f"  drop={trial.params.get('dropout'):.3f}"
              f"  lr={trial.params.get('lr'):.5f}"
              f"  bs={trial.params.get('batch_size')}")
    elif trial.state == optuna.trial.TrialState.PRUNED:
        print(f"  Trial {trial_count[0]:3d}/{N_TRIALS}  PRUNED")

print(f"\nRunning {N_TRIALS} Optuna trials ...\n")
study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])

# ── Leaderboard ────────────────────────────────────────────────
rows = []
for t in study.trials:
    if t.state != optuna.trial.TrialState.COMPLETE:
        continue
    rows.append({
        "trial"      : t.number + 1,
        "val_r2"     : round(t.value, 4),
        "seq_len"    : t.params.get("seq_len"),
        "hidden_size": t.params.get("hidden_size"),
        "num_layers" : t.params.get("num_layers"),
        "dropout"    : round(t.params.get("dropout", 0), 3),
        "lr"         : round(t.params.get("lr", 0), 5),
        "batch_size" : t.params.get("batch_size"),
        "weight_decay": round(t.params.get("weight_decay", 0), 6),
        "epochs"     : t.user_attrs.get("epochs_run"),
    })

res_df = pd.DataFrame(rows).sort_values("val_r2", ascending=False)
print("\n" + "=" * 80)
print(f"LEADERBOARD  GRU weekly  latent_dim={LATENT_DIM}  (top 10)")
print("=" * 80)
print(res_df.head(10).to_string(index=False))
res_df.to_csv(os.path.join(HERE, "optuna_results_gru_weekly.csv"), index=False)

# ── Save best config ───────────────────────────────────────────
best = study.best_trial
cfg  = os.path.join(HERE, "best_config_gru_weekly.txt")
with open(cfg, "w") as f:
    f.write(f"BEST GRU CONFIG  latent_dim={LATENT_DIM}  global tuning\n")
    f.write("=" * 40 + "\n")
    f.write(f"val_r2       : {best.value:.4f}\n")
    f.write(f"n_feats      : {N_FEATS}\n")
    f.write(f"n_provinces  : {N_PROVINCES}\n")
    f.write(f"prov_emb_dim : {PROV_EMB_DIM}\n")
    for k, v in best.params.items():
        f.write(f"{k:16s}: {v}\n")
    f.write("\nNOTE: test data 2022-2023 was never used during tuning.\n")

print(f"\nBest config saved to best_config_gru_weekly.txt")
print(f"Best val_r2 = {best.value:.4f}")
print(f"\nNext step: run  python3 weekly-pipeline/run_gru_weekly.py")
