"""
train_weekly_autoencoder.py
────────────────────────────────────────────────────────────────
Full training run for the weekly autoencoder.
Trains one model for each latent dim in LATENT_DIMS (2 to 9).

Input: one (province, year, week) row = 31 features
       + sin(2pi*week/52) + cos(2pi*week/52) = 33 dims total.

INSTRUCTIONS:
  1. Run tune_weekly_autoencoder.py first.
  2. Copy the printed best config values into the CONFIG block.
  3. Run: python3 weekly-pipeline/train_weekly_autoencoder.py

Outputs for each latent dim K in LATENT_DIMS:
  weekly-pipeline/models/best_model_weekly_dimK.pt
  weekly-pipeline/latents/latent_weekly_dimK.csv
  weekly-pipeline/plots/training_loss_weekly_dimK.png
  weekly-pipeline/plots/latent_pairs_weekly_dimK.png

The latent CSV includes all train and val rows (test locked).
Columns: z1..zK, province, year, week, split
────────────────────────────────────────────────────────────────
"""

import os, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, "extended_input_normalized.csv")

MODELS_DIR = os.path.join(HERE, "models")
LATENT_DIR = os.path.join(HERE, "latents")
PLOTS_DIR  = os.path.join(HERE, "plots")
for d in [MODELS_DIR, LATENT_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── CONFIG: fill in from best_config_weekly.txt after tuning ──
LATENT_DIMS     = [2, 3, 4, 5, 6, 7, 8, 9]   # train all of these
EPOCHS          = 500
PATIENCE        = 80

# Best config from weekly Optuna tuning, val_r2=0.8696
HIDDEN_DIMS     = [512, 256]
USE_BATCHNORM   = True
LR              = 0.00218
DROPOUT         = 0.158
WEIGHT_DECAY    = 1e-5
BATCH_SIZE      = 64
ACTIVATION      = "leakyrelu"
MSE_WEIGHT      = 0.260
OPTIMIZER_NAME  = "adamw"
# ─────────────────────────────────────────────────────────────

TRAIN_YEARS = list(range(2015, 2020))
VAL_YEARS   = [2020, 2021]
TEST_YEARS  = [2022, 2023]

# ── Load and prepare weekly data ───────────────────────────────
print("Loading weekly data ...")
df = pd.read_csv(DATA_CSV)
ID_COLS      = ["province", "year", "month", "week"]
FEATURE_COLS = [c for c in df.columns if c not in ID_COLS]

df["time_sin"] = np.sin(2.0 * np.pi * df["week"] / 52.0)
df["time_cos"] = np.cos(2.0 * np.pi * df["week"] / 52.0)
INPUT_FEATURES = FEATURE_COLS + ["time_sin", "time_cos"]
INPUT_DIM      = len(INPUT_FEATURES)

df = df.dropna(subset=INPUT_FEATURES, how="all")
train_mask = df["year"].isin(TRAIN_YEARS)
val_mask   = df["year"].isin(VAL_YEARS)
trainval_mask = train_mask | val_mask

col_means = df.loc[train_mask, INPUT_FEATURES].mean()
df[INPUT_FEATURES] = df[INPUT_FEATURES].fillna(col_means)

X_train    = torch.tensor(df.loc[train_mask,    INPUT_FEATURES].values.astype(np.float32))
X_val      = torch.tensor(df.loc[val_mask,      INPUT_FEATURES].values.astype(np.float32))
X_trainval = torch.tensor(df.loc[trainval_mask, INPUT_FEATURES].values.astype(np.float32))

meta_trainval = df.loc[trainval_mask, ["province", "year", "week"]].reset_index(drop=True)
meta_trainval["split"] = meta_trainval["year"].apply(
    lambda y: "train" if y in TRAIN_YEARS else "val")

print(f"  Input dim : {INPUT_DIM}  (features + time_sin + time_cos)")
print(f"  Train rows: {len(X_train)}   Val rows: {len(X_val)}  (test locked)")
print(f"  Training latent dims: {LATENT_DIMS}\n")

# ── Model definition ───────────────────────────────────────────
ACTS = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "elu": nn.ELU}

class WeeklyAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims,
                 dropout, activation, use_batchnorm):
        super().__init__()
        act = ACTS[activation]
        enc_dims   = [input_dim] + hidden_dims + [latent_dim]
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

def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1.0 - ss_res / (ss_tot + 1e-8)

def combined_loss(x, x_hat):
    mse = nn.functional.mse_loss(x_hat, x)
    r2  = r2_score(x, x_hat)
    return MSE_WEIGHT * mse + (1.0 - MSE_WEIGHT) * (1.0 - r2)

# ── Train one model per latent dim ─────────────────────────────
summary_rows = []

for latent_dim in LATENT_DIMS:
    print("=" * 60)
    print(f"  Training latent_dim={latent_dim} ...")
    print("=" * 60)

    model = WeeklyAutoencoder(INPUT_DIM, latent_dim, HIDDEN_DIMS,
                               DROPOUT, ACTIVATION, USE_BATCHNORM)

    if OPTIMIZER_NAME == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-6)

    train_loader = DataLoader(TensorDataset(X_train, X_train),
                              batch_size=BATCH_SIZE, shuffle=True)

    model_path = os.path.join(MODELS_DIR, f"best_model_weekly_dim{latent_dim}.pt")
    train_losses, val_losses = [], []
    best_val_loss, patience_count = float("inf"), 0

    n_params = sum(p.numel() for p in model.parameters())
    arch     = [INPUT_DIM] + HIDDEN_DIMS + [latent_dim]
    print(f"  Params: {n_params:,}   Architecture: {arch}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_losses = []
        for xb, _ in train_loader:
            optimizer.zero_grad()
            loss = combined_loss(xb, model(xb))
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = float(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            val_loss = combined_loss(X_val, model(X_val)).item()

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 50 == 0 or epoch == 1:
            with torch.no_grad():
                val_r2 = r2_score(X_val, model(X_val)).item()
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:4d}  train={train_loss:.5f}  val={val_loss:.5f}"
                  f"  val_r2={val_r2:.4f}  lr={cur_lr:.6f}"
                  + ("  *best*" if patience_count == 0 else ""))

        if patience_count >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}.")
            break

    # Load best checkpoint
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        final_r2  = r2_score(X_val, model(X_val)).item()
        final_mse = nn.functional.mse_loss(model(X_val), X_val).item()

    print(f"\n  Final  val_r2={final_r2:.4f}  val_mse={final_mse:.6f}")
    summary_rows.append({"latent_dim": latent_dim,
                          "val_r2": round(final_r2, 4),
                          "val_mse": round(final_mse, 6),
                          "n_params": n_params})

    # ── Save latent representations (train + val only) ─────────
    with torch.no_grad():
        Z = model.encode(X_trainval).numpy()

    z_cols   = [f"z{i+1}" for i in range(latent_dim)]
    latent_df = pd.DataFrame(Z, columns=z_cols)
    latent_df["province"] = meta_trainval["province"].values
    latent_df["year"]     = meta_trainval["year"].values
    latent_df["week"]     = meta_trainval["week"].values
    latent_df["split"]    = meta_trainval["split"].values

    latent_path = os.path.join(LATENT_DIR, f"latent_weekly_dim{latent_dim}.csv")
    latent_df.to_csv(latent_path, index=False)
    print(f"  Saved latent_weekly_dim{latent_dim}.csv  ({len(latent_df)} rows)")

    # ── Training loss plot ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses,   label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Combined Loss")
    ax.set_title(f"Weekly AE  latent_dim={latent_dim}  val_r2={final_r2:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,
                f"training_loss_weekly_dim{latent_dim}.png"), dpi=150)
    plt.close()

    # ── Latent pairs plot (sample up to 6 dims) ─────────────────
    plot_cols = z_cols[:min(latent_dim, 6)]
    pairs     = list(itertools.combinations(plot_cols, 2))
    if pairs:
        ncols = min(3, len(pairs))
        nrows = (len(pairs) + ncols - 1) // ncols
        years = sorted(latent_df["year"].unique())
        yc    = {y: plt.cm.plasma(i / max(len(years) - 1, 1))
                 for i, y in enumerate(years)}
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten() if len(pairs) > 1 else [axes]
        for idx, (za, zb) in enumerate(pairs):
            ax = axes[idx]
            for yr in years:
                m = latent_df["year"] == yr
                ax.scatter(latent_df.loc[m, za], latent_df.loc[m, zb],
                           color=yc[yr], s=4, alpha=0.4, label=str(yr))
            ax.set_xlabel(za, fontsize=8)
            ax.set_ylabel(zb, fontsize=8)
            ax.set_title(f"{za} vs {zb}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(title="Year", fontsize=5, markerscale=2)
        for idx in range(len(pairs), len(axes)):
            axes[idx].set_visible(False)
        plt.suptitle(f"Weekly AE  latent_dim={latent_dim}  val_r2={final_r2:.4f}",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR,
                    f"latent_pairs_weekly_dim{latent_dim}.png"), dpi=150)
        plt.close()

    print(f"  Saved plots for dim {latent_dim}")
    print()

# ── Summary across all dims ────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY  weekly autoencoder  all latent dims")
print("=" * 60)
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(HERE, "summary_weekly_dims.csv"), index=False)

best_row = summary_df.loc[summary_df["val_r2"].idxmax()]
print(f"\n  Best dim by val_r2:  latent_dim={int(best_row['latent_dim'])}"
      f"  val_r2={best_row['val_r2']:.4f}")
print(f"\n  Models saved to: weekly-pipeline/models/")
print(f"  Latents saved to: weekly-pipeline/latents/")
print(f"  Plots saved to: weekly-pipeline/plots/")
print(f"\n  Test data is LOCKED. Next step: run ML benchmark.")
print("─" * 60)
