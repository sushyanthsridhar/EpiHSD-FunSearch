"""
train_autoencoder.py
────────────────────────────────────────────────────────────────
Trains a SINDy Autoencoder on the DR dengue dataset.

WHAT THIS SCRIPT DOES (in plain terms):
  1. Loads extended_input_normalized.csv
  2. Reshapes it into 288 chunks (32 provinces x 9 years)
     Each chunk is one province's full outbreak year: 52 weeks x 31 features
  3. Splits into train (2015-2019), val (2020-2021), test (2022-2023)
  4. Trains an autoencoder that compresses each chunk into 3 numbers
     and then reconstructs it back
  5. Saves the trained model
  6. Extracts the 3-number latent representation for every chunk
  7. Saves latent_representations_latent3.csv (288 rows x 3 columns)
  8. Plots training loss curve
  9. Plots a 3D scatter of all 288 latent points

HOW THE AUTOENCODER WORKS:
  Input  : 1,612 numbers (52 weeks x 31 features, flattened)
  Encoder: 1612 -> 256 -> 64 -> 3   (compression funnel)
  Latent : 3 numbers
  Decoder: 3 -> 64 -> 256 -> 1612  (reconstruction funnel)
  Output : 1,612 numbers (reconstruction of the input)
  Loss   : Mean squared error between input and reconstruction

INSTALL REQUIREMENTS (once):
  pip install torch pandas numpy matplotlib scikit-learn
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
HERE       = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(HERE)
DATA_CSV   = os.path.join(ROOT, "extended_input_normalized.csv")
OUT_DIR    = HERE
os.makedirs(OUT_DIR, exist_ok=True)

# ── Settings (best config from Optuna BO, trial 50) ────────────
# Highest val_r2 = 0.8698. Test data never seen. Ran full 400 epochs
# so 800 epochs gives it more room to improve further.
LATENT_DIM     = 3
EPOCHS         = 800
LR             = 0.00018
BATCH_SIZE     = 16
DROPOUT        = 0.09
MSE_WEIGHT     = 0.25     # 25% MSE, 75% R squared
PATIENCE       = 120      # slow lr needs more patience
WEEKS_PER_YEAR = 52

# Architecture: 1612 -> 256 -> 128 -> 16 -> 3 (encoder)
#               3 -> 16 -> 128 -> 256 -> 1612 (decoder)
HIDDEN_DIMS    = [256, 128]
PRE_BOTTLENECK = 16
ACTIVATION     = "elu"

TRAIN_YEARS = list(range(2015, 2020))   # 5 years x 32 provinces = 160 chunks
VAL_YEARS   = [2020, 2021]              # 2 years x 32 provinces = 64 chunks
TEST_YEARS  = [2022, 2023]              # 2 years x 32 provinces = 64 chunks

# ── Step 1. Load data ──────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(DATA_CSV)

ID_COLS      = ["province", "year", "month", "week"]
FEATURE_COLS = [c for c in df.columns if c not in ID_COLS]
N_FEATURES   = len(FEATURE_COLS)
INPUT_DIM    = WEEKS_PER_YEAR * N_FEATURES   # 52 x 31 = 1,612

print(f"  Features per week : {N_FEATURES}")
print(f"  Input dimension   : {INPUT_DIM}  (52 weeks x {N_FEATURES} features)")

# ── Step 2. Reshape into province-year chunks ──────────────────
print("Reshaping into province-year chunks ...")
df = df.sort_values(["province", "year", "week"]).reset_index(drop=True)

chunks, labels = [], []

for (province, year), grp in df.groupby(["province", "year"]):
    grp_sorted = grp.sort_values("week")
    if len(grp_sorted) < WEEKS_PER_YEAR:
        continue   # skip incomplete years
    chunk = grp_sorted[FEATURE_COLS].values[:WEEKS_PER_YEAR]   # 52 x 31
    chunks.append(chunk.flatten())  # 1,612 numbers
    labels.append({"province": province, "year": year})

chunks = np.array(chunks, dtype=np.float32)
print(f"  Total chunks: {len(chunks)}  (shape: {chunks.shape})")

# Check for NaN values and fill with column mean (0.0 fallback if all-NaN column)
nan_count = np.isnan(chunks).sum()
if nan_count > 0:
    print(f"  WARNING: {nan_count} NaN values found in chunks. Filling with column means.")
    col_means = np.nanmean(chunks, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask  = np.isnan(chunks)
    chunks[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
else:
    print("  No NaN values found. Data is clean.")

# ── Step 3. Train / val split (test data never loaded here) ───
# Test years 2022-2023 are completely locked until final evaluation.
train_idx = [i for i, l in enumerate(labels) if l["year"] in TRAIN_YEARS]
val_idx   = [i for i, l in enumerate(labels) if l["year"] in VAL_YEARS]

X_train = torch.tensor(chunks[train_idx])
X_val   = torch.tensor(chunks[val_idx])

print(f"  Train: {len(X_train)} chunks  Val: {len(X_val)}")
print(f"  Test (2022-2023): locked, not loaded.")

train_loader = DataLoader(TensorDataset(X_train, X_train),
                          batch_size=BATCH_SIZE, shuffle=True)

# ── Step 4. Define the autoencoder ────────────────────────────
ACTS = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "elu": nn.ELU}

class DengueAutoencoder(nn.Module):
    """
    Optimised architecture from Optuna Bayesian search (trial 31).
    Encoder: 1612 -> 512 -> 32 -> 3
    Decoder: 3 -> 32 -> 512 -> 1612
    ELU activation, pre-bottleneck layer at 32 for a gentler final squeeze.
    """
    def __init__(self, input_dim, latent_dim, hidden_dims,
                 pre_bottleneck, dropout, activation):
        super().__init__()
        act = ACTS[activation]

        # Build encoder dims: input -> hidden layers -> pre_bottleneck -> latent
        enc_dims = [input_dim] + hidden_dims
        if pre_bottleneck:
            enc_dims += [pre_bottleneck]
        enc_dims += [latent_dim]

        enc_layers = []
        for i in range(len(enc_dims) - 1):
            enc_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i < len(enc_dims) - 2:     # no activation on the latent layer
                enc_layers.append(act())
                enc_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder mirrors the encoder exactly
        dec_dims = list(reversed(enc_dims))
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec_layers.append(act())
                dec_layers.append(nn.Dropout(dropout))
            else:
                dec_layers.append(nn.Sigmoid())   # keeps output in 0-1
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)

def r2_score(y_true, y_pred):
    """
    R squared. Measures how much variance the reconstruction explains.
    1.0 = perfect. 0.0 = no better than predicting the mean. Negative = worse.
    """
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
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel built. Total parameters: {total_params:,}")
print(f"Architecture: {INPUT_DIM} -> {HIDDEN_DIMS} -> {PRE_BOTTLENECK} -> {LATENT_DIM} (encoder)")

# ── Step 5. Train ──────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs ...")
train_losses, val_losses = [], []

best_val_loss  = float("inf")
patience_count = 0

for epoch in range(1, EPOCHS + 1):
    # Training
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

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = combined_loss(X_val, model(X_val)).item()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(),
                   os.path.join(OUT_DIR, "best_model_latent3.pt"))
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
        print(f"\n  Early stop at epoch {epoch} (no val improvement for {PATIENCE} epochs).")
        break

# Load best model
model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model_latent3.pt")))
model.eval()

# Final scores on train and val only. Test is locked until final evaluation.
with torch.no_grad():
    val_pred  = model(X_val)
    val_loss  = combined_loss(X_val, val_pred).item()
    val_mse   = nn.functional.mse_loss(val_pred, X_val).item()
    val_r2    = r2_score(X_val, val_pred).item()
print(f"\nBest val combined loss : {best_val_loss:.5f}")
print(f"Val MSE                : {val_mse:.5f}  (0 = perfect)")
print(f"Val R squared          : {val_r2:.4f}   (1 = perfect)")
print(f"\nTest data is LOCKED. It will only be used for final evaluation after SINDy is complete.")

# ── Step 6. Extract latent representations (train + val only) ──
# Test years are NOT encoded here. They are held out until final evaluation.
trainval_idx = [i for i, l in enumerate(labels) if l["year"] in TRAIN_YEARS + VAL_YEARS]
trainval_X   = torch.tensor(chunks[trainval_idx])

print(f"\nExtracting latent representations for train+val only ({len(trainval_idx)} chunks) ...")

with torch.no_grad():
    trainval_Z = model.encode(trainval_X).numpy()

latent_df = pd.DataFrame(trainval_Z, columns=["z1", "z2", "z3"])
latent_df["province"] = [labels[i]["province"] for i in trainval_idx]
latent_df["year"]     = [labels[i]["year"]     for i in trainval_idx]
latent_df["split"]    = ["train" if labels[i]["year"] in TRAIN_YEARS
                          else "val" for i in trainval_idx]

latent_df.to_csv(os.path.join(OUT_DIR, "latent_representations_latent3.csv"), index=False)
print(f"Saved latent_representations_latent3.csv  ({len(latent_df)} rows x 3 latent dims)")
print(f"  train={len(latent_df[latent_df.split=='train'])}  val={len(latent_df[latent_df.split=='val'])}")

# ── Step 7. Plot training loss ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_losses, label="Train loss")
ax.plot(val_losses,   label="Val loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Autoencoder Training Loss")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_loss_latent3.png"), dpi=150)
plt.close()
print("Saved training_loss_latent3.png")

# ── Step 8. 3D scatter plot of latent space ────────────────────
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection="3d")

colors = plt.cm.tab20.colors
provinces = sorted(latent_df["province"].unique())
prov_color = {p: colors[i % len(colors)] for i, p in enumerate(provinces)}

for province in provinces:
    mask = latent_df["province"] == province
    pts  = latent_df[mask]
    ax.scatter(pts["z1"], pts["z2"], pts["z3"],
               color=prov_color[province],
               s=40, alpha=0.8, label=province[:20])

ax.set_xlabel("z1  (urban scale and connectivity)")
ax.set_ylabel("z2  (interior vs coastal,  climate curve shape)")
ax.set_zlabel("z3  (epidemic intensity and regional geography)")
ax.set_title("Dengue Outbreak Shape Space\n288 Province-Year Episodes in 3D Latent Space")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "latent_3d_by_province_latent3.png"), dpi=150)
plt.close()
print("Saved latent_3d_by_province_latent3.png")

# Color by year
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection="3d")
years = sorted(latent_df["year"].unique())
year_colors = plt.cm.plasma(np.linspace(0, 1, len(years)))
year_color  = {y: year_colors[i] for i, y in enumerate(years)}

for year in years:
    mask = latent_df["year"] == year
    pts  = latent_df[mask]
    ax.scatter(pts["z1"], pts["z2"], pts["z3"],
               color=year_color[year], s=40, alpha=0.8, label=str(year))

ax.set_xlabel("z1  (urban scale and connectivity)")
ax.set_ylabel("z2  (interior vs coastal,  climate curve shape)")
ax.set_zlabel("z3  (epidemic intensity and regional geography)")
ax.set_title("Dengue Outbreak Shape Space\nColored by Year")
ax.legend(title="Year", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "latent_3d_by_year_latent3.png"), dpi=150)
plt.close()
print("Saved latent_3d_by_year_latent3.png")

print("\n── DONE ─────────────────────────────────────────────────")
print("Files saved in autoencoder/:")
print("  best_model_latent3.pt                 trained autoencoder weights")
print("  latent_representations_latent3.csv    288 province-year points in 3D space")
print("  training_loss_latent3.png             loss curve")
print("  latent_3d_by_province_latent3.png     3D scatter colored by province")
print("  latent_3d_by_year_latent3.png         3D scatter colored by year")
print("\nNext step: run sindy on latent_representations_latent3.csv")
