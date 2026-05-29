"""
train_autoencoder_eval.py
────────────────────────────────────────────────────────────────
EVALUATION COPY of train_autoencoder.py.

This script is identical to the main training script EXCEPT:
  - It loads and evaluates on the test set (2022-2023) at the end.
  - It encodes ALL 9 years (2015-2023) and saves latent_all_years.csv
    which is used by the 3D animation script.

DO NOT use this script during development or hyperparameter tuning.
Only run it when you are ready for a final honest evaluation.

Run:
  python3 train_autoencoder_eval.py
────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Paths ──────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, "extended_input_normalized.csv")
OUT_DIR  = HERE
os.makedirs(OUT_DIR, exist_ok=True)

# ── Settings (same as train_autoencoder.py) ────────────────────
LATENT_DIM     = 3
EPOCHS         = 800
LR             = 0.00018
BATCH_SIZE     = 16
DROPOUT        = 0.09
MSE_WEIGHT     = 0.25
PATIENCE       = 120
WEEKS_PER_YEAR = 52

HIDDEN_DIMS    = [256, 128]
PRE_BOTTLENECK = 16
ACTIVATION     = "elu"

TRAIN_YEARS = list(range(2015, 2020))
VAL_YEARS   = [2020, 2021]
TEST_YEARS  = [2022, 2023]
ALL_YEARS   = list(range(2015, 2024))

# ── Step 1. Load data ──────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(DATA_CSV)

ID_COLS      = ["province", "year", "month", "week"]
FEATURE_COLS = [c for c in df.columns if c not in ID_COLS]
N_FEATURES   = len(FEATURE_COLS)
INPUT_DIM    = WEEKS_PER_YEAR * N_FEATURES

print(f"  Features per week : {N_FEATURES}")
print(f"  Input dimension   : {INPUT_DIM}")

# ── Step 2. Reshape into province-year chunks ──────────────────
print("Reshaping into province-year chunks ...")
df = df.sort_values(["province", "year", "week"]).reset_index(drop=True)

chunks, labels = [], []
for (province, year), grp in df.groupby(["province", "year"]):
    grp_sorted = grp.sort_values("week")
    if len(grp_sorted) < WEEKS_PER_YEAR:
        continue
    chunk = grp_sorted[FEATURE_COLS].values[:WEEKS_PER_YEAR]
    chunks.append(chunk.flatten())
    labels.append({"province": province, "year": year})

chunks = np.array(chunks, dtype=np.float32)
print(f"  Total chunks: {len(chunks)}  (shape: {chunks.shape})")

nan_count = np.isnan(chunks).sum()
if nan_count > 0:
    print(f"  WARNING: {nan_count} NaN values found. Filling with column means.")
    col_means = np.nanmean(chunks, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask  = np.isnan(chunks)
    chunks[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

# ── Step 3. All splits including test ─────────────────────────
train_idx = [i for i, l in enumerate(labels) if l["year"] in TRAIN_YEARS]
val_idx   = [i for i, l in enumerate(labels) if l["year"] in VAL_YEARS]
test_idx  = [i for i, l in enumerate(labels) if l["year"] in TEST_YEARS]

X_train = torch.tensor(chunks[train_idx])
X_val   = torch.tensor(chunks[val_idx])
X_test  = torch.tensor(chunks[test_idx])

print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

train_loader = DataLoader(TensorDataset(X_train, X_train),
                          batch_size=BATCH_SIZE, shuffle=True)

# ── Step 4. Define autoencoder ─────────────────────────────────
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

def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1.0 - ss_res / (ss_tot + 1e-8)

def combined_loss(x, x_hat):
    mse = nn.functional.mse_loss(x_hat, x)
    r2  = r2_score(x, x_hat)
    return MSE_WEIGHT * mse + (1.0 - MSE_WEIGHT) * (1.0 - r2)

model = DengueAutoencoder(
    input_dim      = INPUT_DIM,
    latent_dim     = LATENT_DIM,
    hidden_dims    = HIDDEN_DIMS,
    pre_bottleneck = PRE_BOTTLENECK,
    dropout        = DROPOUT,
    activation     = ACTIVATION,
)
optimizer    = torch.optim.Adam(model.parameters(), lr=LR)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel built. Parameters: {total_params:,}")
print(f"Architecture: {INPUT_DIM} -> {HIDDEN_DIMS} -> {PRE_BOTTLENECK} -> {LATENT_DIM}")

# ── Step 5. Train ──────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs ...")
train_losses, val_losses = [], []
best_val_loss  = float("inf")
patience_count = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    batch_losses = []
    for x_batch, _ in train_loader:
        optimizer.zero_grad()
        x_hat = model(x_batch)
        loss  = combined_loss(x_batch, x_hat)
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
        torch.save(model.state_dict(),
                   os.path.join(OUT_DIR, "best_model_eval.pt"))
        patience_count = 0
    else:
        patience_count += 1

    if epoch % 50 == 0 or epoch == 1:
        with torch.no_grad():
            val_pred = model(X_val)
            val_mse  = nn.functional.mse_loss(val_pred, X_val).item()
            val_r2   = r2_score(X_val, val_pred).item()
        print(f"  Epoch {epoch:4d}  train={train_loss:.5f}  val={val_loss:.5f}"
              f"  val_mse={val_mse:.5f}  val_r2={val_r2:.4f}"
              + ("  *best*" if patience_count == 0 else ""))

    if patience_count >= PATIENCE:
        print(f"\n  Early stop at epoch {epoch}.")
        break

# ── Step 6. Load best and evaluate all splits ──────────────────
model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model_eval.pt")))
model.eval()

print("\n" + "=" * 55)
print("FINAL EVALUATION")
print("=" * 55)
with torch.no_grad():
    for name, X in [("TRAIN", X_train), ("VAL  ", X_val), ("TEST ", X_test)]:
        pred = model(X)
        mse  = nn.functional.mse_loss(pred, X).item()
        r2   = r2_score(X, pred).item()
        print(f"  {name}  MSE={mse:.5f}  R2={r2:.4f}  n={len(X)}")
print("=" * 55)
print("NOTE: test result above is the first and only time test data was used.")

# ── Step 7. Encode ALL 9 years for animation ───────────────────
print("\nEncoding all 9 years for animation ...")
all_X = torch.tensor(chunks)

with torch.no_grad():
    all_Z = model.encode(all_X).numpy()

all_latent_df = pd.DataFrame(all_Z, columns=["z1", "z2", "z3"])
all_latent_df["province"] = [l["province"] for l in labels]
all_latent_df["year"]     = [l["year"]     for l in labels]
all_latent_df["split"]    = ["train" if l["year"] in TRAIN_YEARS
                              else "val"  if l["year"] in VAL_YEARS
                              else "test" for l in labels]

all_latent_df.to_csv(os.path.join(OUT_DIR, "latent_all_years.csv"), index=False)
print(f"Saved latent_all_years.csv  ({len(all_latent_df)} rows, all 9 years)")

# ── Step 8. Training loss plot ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_losses, label="Train")
ax.plot(val_losses,   label="Val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Combined Loss")
ax.set_title("Autoencoder Training Loss (Eval Run)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_loss_eval.png"), dpi=150)
plt.close()
print("Saved training_loss_eval.png")

print("\n── DONE ──────────────────────────────────────────────────")
print("Next step: python3 visualize_latent_animation.py")
