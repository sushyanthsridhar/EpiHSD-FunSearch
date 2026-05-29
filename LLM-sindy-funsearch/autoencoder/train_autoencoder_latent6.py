"""
train_autoencoder_latent6.py
────────────────────────────────────────────────────────────────
Best config from tune_hyperparams_latent6.py  (val_r2 = 0.8888)
  hidden_dims    : [256, 512]
  pre_bottleneck : 32
  shallow_decoder: False
  use_batchnorm  : True
  lr             : 0.002293
  dropout        : 0.0629
  weight_decay   : 0.000003
  batch_size     : 64
  activation     : elu
  mse_weight     : 0.5298

Run:
  python3 train_autoencoder_latent6.py
────────────────────────────────────────────────────────────────
"""

import os, itertools
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, "extended_input_normalized.csv")
OUT_DIR  = HERE

LATENT_DIM      = 6
EPOCHS          = 800
PATIENCE        = 120
WEEKS_PER_YEAR  = 52

HIDDEN_DIMS     = [256, 512]
PRE_BOTTLENECK  = 32
SHALLOW_DECODER = False
USE_BATCHNORM   = True
LR              = 0.002293
DROPOUT         = 0.0629
WEIGHT_DECAY    = 0.000003
BATCH_SIZE      = 64
ACTIVATION      = "elu"
MSE_WEIGHT      = 0.5298

TRAIN_YEARS = list(range(2015, 2020))
VAL_YEARS   = [2020, 2021]
TEST_YEARS  = [2022, 2023]

print(f"Training autoencoder  LATENT_DIM={LATENT_DIM} ...")
df = pd.read_csv(DATA_CSV)
ID_COLS      = ["province", "year", "month", "week"]
FEATURE_COLS = [c for c in df.columns if c not in ID_COLS]
INPUT_DIM    = WEEKS_PER_YEAR * len(FEATURE_COLS)

df = df.sort_values(["province", "year", "week"]).reset_index(drop=True)
chunks, labels = [], []
for (province, year), grp in df.groupby(["province", "year"]):
    grp_s = grp.sort_values("week")
    if len(grp_s) < WEEKS_PER_YEAR:
        continue
    chunks.append(grp_s[FEATURE_COLS].values[:WEEKS_PER_YEAR].flatten())
    labels.append({"province": province, "year": year})

chunks = np.array(chunks, dtype=np.float32)
if np.isnan(chunks).sum() > 0:
    col_means = np.nanmean(chunks, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask  = np.isnan(chunks)
    chunks[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

train_idx = [i for i, l in enumerate(labels) if l["year"] in TRAIN_YEARS]
val_idx   = [i for i, l in enumerate(labels) if l["year"] in VAL_YEARS]
X_train   = torch.tensor(chunks[train_idx])
X_val     = torch.tensor(chunks[val_idx])
print(f"  Train={len(X_train)}  Val={len(X_val)}  (test locked)")
train_loader = DataLoader(TensorDataset(X_train, X_train),
                          batch_size=BATCH_SIZE, shuffle=True)

ACTS = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "elu": nn.ELU}

class DengueAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, pre_bottleneck,
                 dropout, activation, use_batchnorm, shallow_decoder):
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
            self.decoder = nn.Sequential(nn.Linear(latent_dim, input_dim), nn.Sigmoid())
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

def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1.0 - ss_res / (ss_tot + 1e-8)

def combined_loss(x, x_hat):
    mse = nn.functional.mse_loss(x_hat, x)
    r2  = r2_score(x, x_hat)
    return MSE_WEIGHT * mse + (1.0 - MSE_WEIGHT) * (1.0 - r2)

model = DengueAutoencoder(INPUT_DIM, LATENT_DIM, HIDDEN_DIMS, PRE_BOTTLENECK,
                          DROPOUT, ACTIVATION, USE_BATCHNORM, SHALLOW_DECODER)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

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
    train_loss = np.mean(batch_losses)
    model.eval()
    with torch.no_grad():
        val_loss = combined_loss(X_val, model(X_val)).item()
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model_latent6.pt"))
        patience_count = 0
    else:
        patience_count += 1
    if epoch % 50 == 0 or epoch == 1:
        with torch.no_grad():
            val_r2 = r2_score(X_val, model(X_val)).item()
        print(f"  Epoch {epoch:4d}  train={train_loss:.5f}  val={val_loss:.5f}"
              f"  val_r2={val_r2:.4f}" + ("  *best*" if patience_count == 0 else ""))
    if patience_count >= PATIENCE:
        print(f"\n  Early stop at epoch {epoch}.")
        break

model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model_latent6.pt")))
model.eval()
with torch.no_grad():
    val_r2  = r2_score(X_val, model(X_val)).item()
    val_mse = nn.functional.mse_loss(model(X_val), X_val).item()
print(f"\nFinal  val_r2={val_r2:.4f}  val_mse={val_mse:.5f}")
print("Test data is LOCKED.")

trainval_idx = [i for i, l in enumerate(labels) if l["year"] in TRAIN_YEARS + VAL_YEARS]
with torch.no_grad():
    Z = model.encode(torch.tensor(chunks[trainval_idx])).numpy()

cols = [f"z{i+1}" for i in range(LATENT_DIM)]
latent_df = pd.DataFrame(Z, columns=cols)
latent_df["province"] = [labels[i]["province"] for i in trainval_idx]
latent_df["year"]     = [labels[i]["year"]     for i in trainval_idx]
latent_df["split"]    = ["train" if labels[i]["year"] in TRAIN_YEARS else "val"
                          for i in trainval_idx]
latent_df.to_csv(os.path.join(OUT_DIR, "latent_representations_latent6.csv"), index=False)
print(f"Saved latent_representations_latent6.csv  ({len(latent_df)} rows)")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_losses, label="Train")
ax.plot(val_losses, label="Val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Combined Loss")
ax.set_title(f"Training Loss  latent_dim={LATENT_DIM}  val_r2={val_r2:.4f}")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_loss_latent6.png"), dpi=150)
plt.close()

pairs = list(itertools.combinations(cols, 2))
ncols = 3
nrows = (len(pairs) + ncols - 1) // ncols
years = sorted(latent_df.year.unique())
yc = {y: plt.cm.plasma(i / len(years)) for i, y in enumerate(years)}
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = np.array(axes).flatten()
for idx, (za, zb) in enumerate(pairs):
    ax = axes[idx]
    for yr in years:
        m = latent_df.year == yr
        ax.scatter(latent_df.loc[m, za], latent_df.loc[m, zb],
                   color=yc[yr], s=25, alpha=0.75, label=str(yr))
    ax.set_xlabel(za)
    ax.set_ylabel(zb)
    ax.set_title(f"{za} vs {zb}")
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(title="Year", fontsize=6)
for idx in range(len(pairs), len(axes)):
    axes[idx].set_visible(False)
plt.suptitle(f"Pairwise Latent Dims  latent_dim={LATENT_DIM}  val_r2={val_r2:.4f}", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "latent_pairs_latent6.png"), dpi=150)
plt.close()
print("Saved training_loss_latent6.png  latent_pairs_latent6.png")
print(f"\n── DONE  latent_dim=6 ───────────────────────────────────")
