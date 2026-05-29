"""
tune_weekly_autoencoder.py
────────────────────────────────────────────────────────────────
Bayesian hyperparameter tuning for the weekly autoencoder.

Each training sample is one (province, year, week) observation.
Input: 31 original features + sin(2pi*week/52) + cos(2pi*week/52)
       = 33 dimensions per sample.

This gives ~8320 training samples vs 160 for the annual version,
a 52x increase that makes generalisation viable.

Tuning is run on LATENT_DIM=6 to produce a config that is then
used for all dims 2-9 in train_weekly_autoencoder.py.

Test data (2022-2023) is NEVER loaded during tuning.

Run:
  python3 weekly-pipeline/tune_weekly_autoencoder.py

Outputs:
  weekly-pipeline/optuna_results_weekly.csv
  weekly-pipeline/best_config_weekly.txt
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
    raise ImportError("Run:  pip install optuna  then try again.")

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, "extended_input_normalized.csv")

LATENT_DIM  = 6
N_TRIALS    = 100
MAX_EPOCHS  = 300
PATIENCE    = 50

TRAIN_YEARS = list(range(2015, 2020))
VAL_YEARS   = [2020, 2021]
# 2022, 2023 are LOCKED, never loaded here

# ── Load data ─────────────────────────────────────────────────
print("Loading weekly data ...")
df = pd.read_csv(DATA_CSV)
ID_COLS      = ["province", "year", "month", "week"]
FEATURE_COLS = [c for c in df.columns if c not in ID_COLS]

# Circular time encoding
df["time_sin"] = np.sin(2.0 * np.pi * df["week"] / 52.0)
df["time_cos"] = np.cos(2.0 * np.pi * df["week"] / 52.0)
INPUT_FEATURES = FEATURE_COLS + ["time_sin", "time_cos"]
INPUT_DIM      = len(INPUT_FEATURES)

print(f"  Original features: {len(FEATURE_COLS)}")
print(f"  Input dim after time encoding: {INPUT_DIM}  (added sin, cos of week)")

# Drop rows with all NaNs
df = df.dropna(subset=INPUT_FEATURES, how="all")

# Fill remaining NaNs with column means from train
train_mask = df["year"].isin(TRAIN_YEARS)
col_means  = df.loc[train_mask, INPUT_FEATURES].mean()
df[INPUT_FEATURES] = df[INPUT_FEATURES].fillna(col_means)

X_train = torch.tensor(
    df.loc[train_mask, INPUT_FEATURES].values.astype(np.float32))
X_val   = torch.tensor(
    df.loc[df["year"].isin(VAL_YEARS), INPUT_FEATURES].values.astype(np.float32))

print(f"  Train samples: {len(X_train)}  Val samples: {len(X_val)}  (test locked)")
print(f"  Latent dim for tuning: {LATENT_DIM}")
print(f"  Running {N_TRIALS} Optuna trials ...\n")

# ── Loss helpers ───────────────────────────────────────────────
def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return (1.0 - ss_res / (ss_tot + 1e-8)).item()

def combined_loss(x, x_hat, mse_w):
    mse = nn.functional.mse_loss(x_hat, x)
    r2  = r2_score(x, x_hat)
    return mse_w * mse + (1.0 - mse_w) * (1.0 - r2)

# ── Model ──────────────────────────────────────────────────────
ACTIVATIONS = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "elu": nn.ELU}

class WeeklyAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims,
                 dropout, activation, use_batchnorm):
        super().__init__()
        act = ACTIVATIONS[activation]

        enc_dims    = [input_dim] + hidden_dims + [latent_dim]
        enc_layers  = []
        for i in range(len(enc_dims) - 1):
            enc_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i < len(enc_dims) - 2:
                if use_batchnorm:
                    enc_layers.append(nn.BatchNorm1d(enc_dims[i + 1]))
                enc_layers.append(act())
                if dropout > 0.0:
                    enc_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*enc_layers)

        dec_dims   = list(reversed(enc_dims))
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

# ── Optuna objective ───────────────────────────────────────────
def objective(trial):
    n_layers   = trial.suggest_int("n_layers", 1, 3)
    WIDTH_POOL = [32, 64, 128, 256, 512]
    hidden_dims = [trial.suggest_categorical(f"h{i}", WIDTH_POOL)
                   for i in range(n_layers)]

    use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])
    lr            = trial.suggest_float("lr",           1e-4, 5e-3, log=True)
    dropout       = trial.suggest_float("dropout",      0.0,  0.35)
    weight_decay  = trial.suggest_float("weight_decay", 0.0,  1e-3)
    batch_size    = trial.suggest_categorical("batch_size", [64, 128, 256])
    activation    = trial.suggest_categorical("activation", ["relu", "leakyrelu", "elu"])
    mse_weight    = trial.suggest_float("mse_weight",   0.2,  0.9)
    optimizer_name= trial.suggest_categorical("optimizer", ["adam", "adamw"])

    if use_batchnorm and batch_size < 16:
        raise optuna.exceptions.TrialPruned()

    model = WeeklyAutoencoder(
        input_dim   = INPUT_DIM,
        latent_dim  = LATENT_DIM,
        hidden_dims = hidden_dims,
        dropout     = dropout,
        activation  = activation,
        use_batchnorm = use_batchnorm,
    )
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = DataLoader(TensorDataset(X_train, X_train),
                        batch_size=batch_size, shuffle=True)

    best_val, best_state, patience_c = float("inf"), None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, _ in loader:
            optimizer.zero_grad()
            loss = combined_loss(xb, model(xb), mse_weight)
            if torch.isnan(loss):
                return float("nan")
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vl = combined_loss(X_val, model(X_val), mse_weight)
            if not isinstance(vl, float):
                vl = vl.item()
            if vl != vl:
                return float("nan")

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1

        if patience_c >= PATIENCE:
            break

        trial.report(vl, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    if best_state is None:
        return float("nan")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_r2  = r2_score(X_val, model(X_val))
        val_mse = nn.functional.mse_loss(model(X_val), X_val).item()

    trial.set_user_attr("val_mse",     round(val_mse, 6))
    trial.set_user_attr("n_params",    sum(p.numel() for p in model.parameters()))
    trial.set_user_attr("hidden_dims", str(hidden_dims))
    trial.set_user_attr("epochs_run",  epoch)

    return val_r2

# ── Run study ─────────────────────────────────────────────────
sampler = optuna.samplers.TPESampler(seed=42)
pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
study   = optuna.create_study(direction="maximize",
                               sampler=sampler, pruner=pruner)

trial_count = [0]
def callback(study, trial):
    trial_count[0] += 1
    if trial.state == optuna.trial.TrialState.COMPLETE:
        print(f"  Trial {trial_count[0]:3d}/{N_TRIALS}"
              f"  val_r2={trial.value:.4f}"
              f"  layers={trial.user_attrs.get('hidden_dims','?')}"
              f"  bn={trial.params.get('use_batchnorm','?')}"
              f"  lr={trial.params.get('lr', 0):.5f}"
              f"  act={trial.params.get('activation','?')}"
              f"  opt={trial.params.get('optimizer','?')}")
    elif trial.state == optuna.trial.TrialState.PRUNED:
        print(f"  Trial {trial_count[0]:3d}/{N_TRIALS}  PRUNED")

study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])

# ── Leaderboard ────────────────────────────────────────────────
print("\n" + "=" * 100)
print(f"LEADERBOARD  weekly autoencoder  latent_dim={LATENT_DIM}  (top 10 by val_r2)")
print("=" * 100)

rows = []
for t in study.trials:
    if t.state != optuna.trial.TrialState.COMPLETE:
        continue
    rows.append({
        "trial":        t.number + 1,
        "val_r2":       round(t.value, 4),
        "val_mse":      t.user_attrs.get("val_mse"),
        "hidden_dims":  t.user_attrs.get("hidden_dims"),
        "use_batchnorm":t.params.get("use_batchnorm"),
        "lr":           round(t.params.get("lr", 0), 5),
        "dropout":      round(t.params.get("dropout", 0), 3),
        "weight_decay": round(t.params.get("weight_decay", 0), 6),
        "batch_size":   t.params.get("batch_size"),
        "activation":   t.params.get("activation"),
        "mse_weight":   round(t.params.get("mse_weight", 0), 3),
        "optimizer":    t.params.get("optimizer"),
        "n_params":     t.user_attrs.get("n_params"),
        "epochs":       t.user_attrs.get("epochs_run"),
    })

res_df = pd.DataFrame(rows).sort_values("val_r2", ascending=False)
print(res_df.head(10).to_string(index=False))

out_csv = os.path.join(HERE, "optuna_results_weekly.csv")
res_df.to_csv(out_csv, index=False)
print(f"\nAll results saved to optuna_results_weekly.csv")

# ── Save best config ───────────────────────────────────────────
best = study.best_trial
cfg_path = os.path.join(HERE, "best_config_weekly.txt")
with open(cfg_path, "w") as f:
    f.write(f"BEST WEEKLY AUTOENCODER CONFIG  latent_dim={LATENT_DIM}\n")
    f.write("=" * 40 + "\n")
    f.write(f"val_r2         : {best.value:.4f}\n")
    f.write(f"input_dim      : {INPUT_DIM}\n")
    f.write(f"hidden_dims    : {best.user_attrs.get('hidden_dims')}\n")
    f.write(f"n_params       : {best.user_attrs.get('n_params')}\n")
    for k, v in best.params.items():
        f.write(f"{k:16s}: {v}\n")
    f.write("\nNOTE: test data 2022-2023 was never used during tuning.\n")

print(f"\nBest config saved to best_config_weekly.txt")
print(f"\nBEST RESULT (val only, test still locked):")
print(f"  val_r2       = {best.value:.4f}")
print(f"  input_dim    = {INPUT_DIM}")
print(f"  hidden_dims  = {best.user_attrs.get('hidden_dims')}")
print(f"  use_batchnorm= {best.params.get('use_batchnorm')}")
print(f"  lr           = {best.params.get('lr'):.5f}")
print(f"  dropout      = {best.params.get('dropout'):.3f}")
print(f"  activation   = {best.params.get('activation')}")
print(f"  batch_size   = {best.params.get('batch_size')}")
print(f"  mse_weight   = {best.params.get('mse_weight'):.3f}")
print(f"  optimizer    = {best.params.get('optimizer')}")
print(f"\nNext step: copy these values into train_weekly_autoencoder.py CONFIG block")
print(f"then run:  python3 weekly-pipeline/train_weekly_autoencoder.py")
