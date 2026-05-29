"""
province_similarity.py
────────────────────────────────────────────────────────────────────────────
Finds what actually separates Island A (EIP-driven) from Island B (AR-driven)
provinces by building a feature matrix of geographic, climatic, and
epidemiological attributes and running:

  1. PCA scatter  — do the two islands separate in feature space?
  2. Feature importance  — which attributes best predict island membership?
  3. Pairwise similarity heatmap  — within-island vs cross-island distances
  4. Printed summary table  — all features side by side with island means

Install:
    pip install pandas numpy matplotlib scikit-learn seaborn

Run:
    cd funsearch/analysis/
    python province_similarity.py
────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import seaborn as sns

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results", "province_breakdown.csv")

# ── Province metadata ─────────────────────────────────────────────────────────
# lat, lon, elevation_m (approximate mean elevation of province),
# dist_coast_km (approximate straight-line distance to nearest coast),
# mean_temp_c   (approximate annual mean temperature),
# urban_index   (1=major urban centre, 0.5=mixed, 0=rural)
# dengue_burden (qualitative annual incidence tier: high=2, medium=1, low=0)

METADATA = {
    # province              lat      lon    elev  d_coast  temp  urban  burden
    "32 Santo Domingo"   : (18.485,-69.931,  30,    2,    27.5,  1.0,   2),
    "01 Distrito Nacional": (18.479,-69.890,  14,    3,    28.0,  1.0,   2),
    "13 La Vega"         : (19.221,-70.529, 110,   60,    25.5,  0.5,   2),
    "04 Barahona"        : (18.211,-71.101,  50,    5,    27.8,  0.5,   1),
    "06 Duarte"          : (19.199,-70.033,  80,   55,    26.0,  0.5,   2),
    "24 Sánchez Ramírez" : (19.052,-70.153, 130,   65,    25.5,  0.0,   1),
    "11 La Altagracia"   : (18.616,-68.712,  20,    8,    28.5,  0.5,   1),
    "22 San Juan"        : (18.806,-71.228, 400,   80,    24.0,  0.5,   1),
    "28 Monseñor Nouel"  : (18.922,-70.415, 200,   70,    25.0,  0.5,   1),
    "31 San José de Ocoa": (18.544,-70.503, 550,   50,    23.0,  0.0,   0),
    "30 Hato Mayor"      : (18.764,-69.255, 140,   35,    26.5,  0.0,   1),
    "20 Samaná"          : (19.206,-69.336,  40,    5,    27.0,  0.0,   1),
    "03 Baoruco"         : (18.486,-71.418, 180,   20,    28.8,  0.0,   0),
    "25 Santiago"        : (19.451,-70.697, 170,   90,    26.5,  1.0,   2),
    "21 San Cristóbal"   : (18.418,-70.106,  80,   10,    27.0,  0.5,   1),
    "18 Puerto Plata"    : (19.795,-70.685,  60,    5,    26.0,  0.5,   1),
    "15 Monte Cristi"    : (19.866,-71.648,  20,    5,    28.0,  0.0,   0),
    "29 Monte Plata"     : (18.806,-69.784, 160,   45,    26.0,  0.0,   1),
    "16 Pedernales"      : (17.929,-71.444, 100,   15,    27.5,  0.0,   0),
    "07 Elías Piña"      : (18.875,-71.706, 600,  110,    22.0,  0.0,   0),
    "02 Azua"            : (18.452,-70.735, 120,   25,    28.0,  0.5,   1),
    "05 Dajabón"         : (19.549,-71.707,  80,   30,    26.5,  0.0,   0),
    "08 El Seibo"        : (18.765,-69.038, 200,   30,    26.0,  0.0,   1),
    "09 Espaillat"       : (19.623,-70.275, 250,   40,    24.5,  0.0,   1),
    "10 Independencia"   : (18.407,-71.845, 700,   30,    22.5,  0.0,   0),
    "12 La Romana"       : (18.427,-68.972,  15,    5,    28.0,  0.5,   1),
    "14 Mª Trinidad Sánchez": (19.374,-69.853, 30,  5,   27.0,  0.0,   1),
    "17 Peravia"         : (18.280,-70.336, 100,   10,    27.5,  0.5,   1),
    "19 Hermanas Mirabal": (19.375,-70.307, 200,   50,    25.0,  0.0,   1),
    "23 San Pedro de Macorís": (18.451,-69.301, 15, 3,   27.5,  0.5,   1),
    "26 Santiago Rodríguez": (19.481,-71.336, 300, 80,   24.5,  0.0,   0),
    "27 Valverde"        : (19.583,-71.072, 100,   60,    26.0,  0.5,   1),
}

ISLAND_A_KEYS = {
    "32 Santo Domingo", "01 Distrito Nacional", "13 La Vega",
    "04 Barahona", "06 Duarte", "24 Sánchez Ramírez",
    "11 La Altagracia", "22 San Juan", "28 Monseñor Nouel",
    "31 San José de Ocoa", "30 Hato Mayor", "20 Samaná", "03 Baoruco",
}

COL_A = "#1565C0"
COL_B = "#2E7D32"


# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════

def load():
    # Geographic + climatic features
    rows = []
    for prov, (lat, lon, elev, d_coast, temp, urban, burden) in METADATA.items():
        island = "A_EIP" if prov in ISLAND_A_KEYS else "B_AR"
        rows.append({
            "province"   : prov,
            "island"     : island,
            "lat"        : lat,
            "lon"        : lon,
            "elevation_m": elev,
            "dist_coast_km": d_coast,
            "mean_temp_c": temp,
            "urban_index": urban,
            "dengue_burden": burden,
        })
    geo_df = pd.DataFrame(rows).set_index("province")

    # R² results from FunSearch run
    if os.path.exists(RESULTS):
        res = pd.read_csv(RESULTS)
        res["province"] = res["province"].str.strip()
        res = res.set_index("province")
        # Rename María Trinidad Sánchez to match metadata key
        if "14 María Trinidad Sánchez" in res.index:
            res = res.rename(index={"14 María Trinidad Sánchez": "14 Mª Trinidad Sánchez"})
        for col in ["fs_r2_test", "fs_r2_val", "fs_r2_train",
                    "fs_spec_test", "sindy_r2_test", "delta_test"]:
            if col in res.columns:
                geo_df[col] = res[col]
    else:
        print(f"WARNING: {RESULTS} not found — skipping R² features")

    return geo_df


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_pca(df, feat_cols, ax):
    X = df[feat_cols].dropna()
    labels = df.loc[X.index, "island"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    Xp  = pca.fit_transform(Xs)

    colors = [COL_A if l == "A_EIP" else COL_B for l in labels]
    markers= ["*"    if l == "A_EIP" else "o"  for l in labels]

    for i, (xi, yi, c, m, prov) in enumerate(
            zip(Xp[:,0], Xp[:,1], colors, markers, labels.index)):
        ax.scatter(xi, yi, c=c, marker=m,
                   s=160 if m=="*" else 80,
                   edgecolors="white", linewidths=1.2, zorder=3)
        short = prov.split(None,1)[1] if " " in prov else prov
        ax.annotate(short, (xi, yi), fontsize=6,
                    xytext=(4,4), textcoords="offset points",
                    color=c, zorder=4)

    pct = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1  ({pct[0]:.1f}% var)", fontsize=9)
    ax.set_ylabel(f"PC2  ({pct[1]:.1f}% var)", fontsize=9)
    ax.set_title("PCA — do the two islands separate in feature space?",
                 fontsize=10, fontweight="bold")
    ax.axhline(0, color="#ddd", lw=0.7)
    ax.axvline(0, color="#ddd", lw=0.7)
    ax.grid(True, alpha=0.3)

    patch_a = mpatches.Patch(color=COL_A, label="Island A — EIP-driven")
    patch_b = mpatches.Patch(color=COL_B, label="Island B — AR-driven")
    ax.legend(handles=[patch_a, patch_b], fontsize=8)

    # Loading arrows
    ax2 = ax.twinx().twiny()
    ax2.set_xlim(-1,1); ax2.set_ylim(-1,1)
    comps = pca.components_
    for j, fname in enumerate(feat_cols):
        ax2.annotate("", xy=(comps[0,j]*0.9, comps[1,j]*0.9), xytext=(0,0),
                     arrowprops=dict(arrowstyle="->", color="#FF6F00", lw=1.2))
        ax2.text(comps[0,j]*0.95, comps[1,j]*0.95, fname,
                 fontsize=6.5, color="#FF6F00", ha="center")
    ax2.set_xticks([]); ax2.set_yticks([])

    return pca, scaler


def plot_importance(df, feat_cols, ax):
    X = df[feat_cols].dropna()
    y = (df.loc[X.index, "island"] == "A_EIP").astype(int)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(Xs, y)
    imp = rf.feature_importances_

    order = np.argsort(imp)
    colors_bar = [COL_A if imp[i] > np.median(imp) else "#AAAAAA"
                  for i in order]
    ax.barh([feat_cols[i] for i in order], imp[order],
            color=colors_bar, edgecolor="white")
    ax.axvline(np.median(imp), color="#FF6F00", lw=1.2, linestyle="--",
               label="median")
    ax.set_xlabel("Feature importance (Random Forest)", fontsize=9)
    ax.set_title("Which features best separate Island A from Island B?",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    return rf


def plot_heatmap(df, feat_cols, ax):
    from sklearn.metrics.pairwise import euclidean_distances
    X = df[feat_cols].dropna()
    labels = df.loc[X.index, "island"]
    short_names = [p.split(None,1)[1] if " " in p else p for p in X.index]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    D  = euclidean_distances(Xs)
    D_df = pd.DataFrame(D, index=short_names, columns=short_names)

    # Sort so Island A comes first
    order = [i for i,l in zip(range(len(labels)),labels) if l=="A_EIP"] + \
            [i for i,l in zip(range(len(labels)),labels) if l!="A_EIP"]
    sorted_names = [short_names[i] for i in order]
    D_sorted = D_df.loc[sorted_names, sorted_names]

    n_a = sum(1 for l in labels if l=="A_EIP")
    sns.heatmap(D_sorted, ax=ax, cmap="YlOrRd_r",
                xticklabels=True, yticklabels=True,
                linewidths=0.2, linecolor="#eeeeee",
                cbar_kws={"label":"Euclidean distance (standardised features)"})
    ax.tick_params(labelsize=6)

    # Draw dividing lines between islands
    ax.axhline(n_a, color="black", lw=2)
    ax.axvline(n_a, color="black", lw=2)
    ax.text(n_a/2,       -0.5, "Island A (EIP)", ha="center",
            fontsize=8, fontweight="bold", color=COL_A,
            transform=ax.get_xaxis_transform())
    ax.text(n_a+(len(labels)-n_a)/2, -0.5, "Island B (AR)", ha="center",
            fontsize=8, fontweight="bold", color=COL_B,
            transform=ax.get_xaxis_transform())
    ax.set_title("Pairwise province similarity\n(darker = more similar)",
                 fontsize=10, fontweight="bold")


def plot_violin(df, feat_cols, axes):
    for ax, feat in zip(axes, feat_cols):
        a_vals = df.loc[df["island"]=="A_EIP", feat].dropna()
        b_vals = df.loc[df["island"]=="B_AR",  feat].dropna()
        parts = ax.violinplot([a_vals, b_vals], positions=[0,1],
                              showmedians=True, showextrema=True)
        for i, (pc, col) in enumerate(zip(parts["bodies"], [COL_A, COL_B])):
            pc.set_facecolor(col); pc.set_alpha(0.65)
        parts["cmedians"].set_color("white")
        ax.set_xticks([0,1])
        ax.set_xticklabels(["A (EIP)", "B (AR)"], fontsize=8)
        ax.set_title(feat, fontsize=8, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Annotate means
        ax.text(0, a_vals.mean(), f"μ={a_vals.mean():.1f}",
                fontsize=7, ha="center", va="bottom", color=COL_A)
        ax.text(1, b_vals.mean(), f"μ={b_vals.mean():.1f}",
                fontsize=7, ha="center", va="bottom", color=COL_B)


# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(df, feat_cols):
    print("\n" + "═"*80)
    print("  ISLAND FEATURE COMPARISON  (mean ± std)")
    print("═"*80)
    print(f"  {'Feature':<25} {'Island A (EIP)':>20} {'Island B (AR)':>20}  {'Δ (A−B)':>10}")
    print("  " + "─"*74)
    a = df[df["island"]=="A_EIP"]
    b = df[df["island"]=="B_AR"]
    for f in feat_cols:
        av = a[f].dropna();  bv = b[f].dropna()
        print(f"  {f:<25} {av.mean():>8.2f} ± {av.std():>5.2f}   "
              f"{bv.mean():>8.2f} ± {bv.std():>5.2f}   "
              f"{av.mean()-bv.mean():>+10.2f}")
    print("═"*80)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df = load()

    # Feature set — geographic + climatic + epidemiological
    geo_feats = ["lat", "lon", "elevation_m", "dist_coast_km",
                 "mean_temp_c", "urban_index", "dengue_burden"]
    r2_feats  = ["fs_r2_test", "fs_r2_val", "sindy_r2_test", "delta_test",
                 "fs_spec_test"]
    r2_feats  = [f for f in r2_feats if f in df.columns]
    all_feats = geo_feats + r2_feats

    print_summary(df, all_feats)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 18), facecolor="white")
    fig.suptitle(
        "Province Similarity Analysis — What actually separates Island A (EIP) from Island B (AR)?",
        fontsize=14, fontweight="bold", y=0.98,
    )

    gs = fig.add_gridspec(3, 3,
                          hspace=0.45, wspace=0.38,
                          left=0.06, right=0.97,
                          top=0.94, bottom=0.05)

    ax_pca  = fig.add_subplot(gs[0, :2])
    ax_imp  = fig.add_subplot(gs[0,  2])
    ax_heat = fig.add_subplot(gs[1, :])

    violin_feats = ["elevation_m", "mean_temp_c", "dist_coast_km",
                    "dengue_burden", "urban_index",
                    "delta_test" if "delta_test" in df.columns else "lat"]
    violin_feats = [f for f in violin_feats if f in df.columns]
    violin_axes  = [fig.add_subplot(gs[2, j]) for j in range(min(3,len(violin_feats)))]
    # fit remaining violins
    extra_axes = []
    if len(violin_feats) > 3:
        gs2 = fig.add_gridspec(3, len(violin_feats)-3,
                               left=0.06, right=0.97,
                               top=0.94, bottom=0.05)

    # ── Run plots ─────────────────────────────────────────────────────────────
    plot_pca(df, all_feats, ax_pca)
    plot_importance(df, all_feats, ax_imp)
    plot_heatmap(df, all_feats, ax_heat)
    plot_violin(df, violin_feats[:3], violin_axes)

    out = os.path.join(HERE, "province_similarity.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nFigure  →  {out}")


if __name__ == "__main__":
    main()
