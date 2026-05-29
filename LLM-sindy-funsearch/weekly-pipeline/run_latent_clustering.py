"""
run_latent_clustering.py
────────────────────────────────────────────────────────────────
Cluster provinces using the annual latent representations.

Strategy:
  1. Load latent_representations_latent6.csv (train+val rows).
  2. Compute one vector per province by averaging z1..z6 over
     all training years (2015-2019) to get a stable fingerprint.
  3. Run K-means for k=2..8, pick best k by silhouette score.
  4. Visualise clusters in 2D PCA space.
  5. Output province_clusters.csv for use in per-cluster ML and SINDy.

Why clustering:
  Province heterogeneity is the main reason globally pooled models
  fail. Provinces like Distrito Nacional and Azua have structurally
  different epidemic dynamics. Cluster-level models allow per-group
  equation discovery. Cluster differences may also point to ghost
  variables (unmeasured confounders) worth discussing in the paper.

Outputs (all in weekly-pipeline/):
  province_clusters.csv          province, cluster_id, z1..z6 mean
  cluster_pca_plot.png           2D PCA coloured by cluster
  cluster_silhouette_plot.png    silhouette score vs k
  cluster_summary.txt            which provinces fall in each cluster
────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
AE_DIR   = os.path.join(ROOT, "autoencoder")

LATENT_DIM  = 6
LATENT_CSV  = os.path.join(AE_DIR, f"latent_representations_latent{LATENT_DIM}.csv")
TRAIN_YEARS = list(range(2015, 2020))
K_RANGE     = range(2, 9)

print(f"Loading annual latent representations (dim={LATENT_DIM}) ...")
latent = pd.read_csv(LATENT_CSV)
z_cols = [f"z{i+1}" for i in range(LATENT_DIM)]

# Keep only training years for computing province fingerprints.
# Using train only avoids any leakage from val dynamics.
train_latent = latent[latent["split"] == "train"].copy()
print(f"  Train rows: {len(train_latent)}  ({len(train_latent['province'].unique())} provinces)")

# One vector per province: mean over training years
province_z = (
    train_latent.groupby("province")[z_cols]
    .mean()
    .reset_index()
)
print(f"  Province fingerprints: {len(province_z)}")

# ── Standardise before clustering ─────────────────────────────
scaler = StandardScaler()
Z = scaler.fit_transform(province_z[z_cols].values)

# ── K-means for k=2..8 ────────────────────────────────────────
print("\n  Silhouette scores:")
sil_scores = {}
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(Z)
    sil = silhouette_score(Z, labels)
    sil_scores[k] = sil
    print(f"  k={k}  silhouette={sil:.4f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"\n  Best k by silhouette: {best_k}  (score={sil_scores[best_k]:.4f})")

# ── Fit final clustering with best k ──────────────────────────
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
province_z["cluster_id"] = km_final.fit_predict(Z)

# ── Cluster summary ───────────────────────────────────────────
summary_lines = [f"LATENT SPACE CLUSTERING  dim={LATENT_DIM}  best_k={best_k}\n",
                 f"Silhouette scores: {sil_scores}\n\n"]
for c in sorted(province_z["cluster_id"].unique()):
    provs = province_z[province_z["cluster_id"] == c]["province"].tolist()
    summary_lines.append(f"Cluster {c}  ({len(provs)} provinces):\n")
    for p in sorted(provs):
        summary_lines.append(f"  {p}\n")
    summary_lines.append("\n")

summary_path = os.path.join(HERE, "cluster_summary.txt")
with open(summary_path, "w") as f:
    f.writelines(summary_lines)

print("\n" + "".join(summary_lines))

# ── Save province cluster assignments ─────────────────────────
out_cols = ["province", "cluster_id"] + z_cols
province_z[out_cols].to_csv(
    os.path.join(HERE, "province_clusters.csv"), index=False
)
print(f"  Saved province_clusters.csv")

# ── Also annotate full latent CSV with cluster ─────────────────
cluster_map = dict(zip(province_z["province"], province_z["cluster_id"]))
latent["cluster_id"] = latent["province"].map(cluster_map)
latent.to_csv(
    os.path.join(AE_DIR, f"latent_representations_latent{LATENT_DIM}_clustered.csv"),
    index=False
)
print(f"  Saved latent_representations_latent{LATENT_DIM}_clustered.csv")

# ── Silhouette score plot ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ks   = list(sil_scores.keys())
vals = list(sil_scores.values())
ax.plot(ks, vals, "o-", color="#4C72B0", linewidth=2)
ax.axvline(best_k, color="red", linestyle="--", linewidth=1, label=f"best k={best_k}")
ax.set_xlabel("Number of clusters k")
ax.set_ylabel("Silhouette score")
ax.set_title(f"K-means silhouette score  latent_dim={LATENT_DIM}")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "cluster_silhouette_plot.png"), dpi=150)
plt.close()

# ── PCA 2D visualisation ───────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
Z_2d = pca.fit_transform(Z)
var_explained = pca.explained_variance_ratio_

colors = plt.cm.tab10(np.linspace(0, 1, best_k))
fig, ax = plt.subplots(figsize=(10, 8))

for c in sorted(province_z["cluster_id"].unique()):
    mask = province_z["cluster_id"].values == c
    ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
               color=colors[c], s=100, zorder=3, label=f"Cluster {c}")
    for idx in np.where(mask)[0]:
        pname = province_z["province"].iloc[idx]
        short = pname.split(" ", 1)[-1] if " " in pname else pname
        ax.annotate(short, (Z_2d[idx, 0], Z_2d[idx, 1]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")
ax.set_title(f"Province clusters in latent space  k={best_k}  dim={LATENT_DIM}")
ax.legend(loc="best", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "cluster_pca_plot.png"), dpi=150)
plt.close()

print(f"  Saved cluster_pca_plot.png")
print(f"  Saved cluster_silhouette_plot.png")
print(f"\n  Done. Next step: run run_ml_benchmark_weekly_v2.py")
print(f"  Then run per-cluster SINDy using province_clusters.csv to group provinces.")
print("─" * 60)
