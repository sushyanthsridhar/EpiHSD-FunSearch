"""
didt_correlations.py
────────────────────────────────────────────────────────────────
Finds what features are most correlated with dI/dt.

dI/dt is the rate of change of dengue incidence week to week.
This is the quantity SINDy will try to model, so understanding
what drives it is critical before building the library.

WHAT THIS DOES:
  1. Pearson correlation of every feature vs dI/dt at weekly level
  2. Spearman correlation (rank-based, catches nonlinear links)
  3. Lag analysis: does feature at week t-1, t-2, t-3 predict dI/dt at t?
  4. Province-year level: which provinces have the highest mean |dI/dt|?
  5. Scatter plots of the top 6 most correlated features vs dI/dt
  6. Heatmap: feature vs dI/dt correlation broken down by year

Run from the project root:
  python3 analysis/didt_correlations.py

Outputs saved to analysis/didt_results/
────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

# ── Paths ──────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, "extended_input_normalized.csv")
OUT_DIR  = os.path.join(HERE, "didt_results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────
DARK_BG  = "#0f0f1a"
PANEL_BG = "#12122a"
TEXT_COL = "#dde1ff"
GRID_COL = "#222244"
ACC1     = "#e63946"
ACC2     = "#2a9d8f"
ACC3     = "#f4a261"

def style_ax(ax, title=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.7)
    if title:
        ax.set_title(title, color=TEXT_COL, fontsize=10, pad=6)

# ── Load ───────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(DATA_CSV)
df = df.sort_values(["province", "year", "week"]).reset_index(drop=True)

ID_COLS      = ["province", "year", "month", "week"]
FEATURE_COLS = [c for c in df.columns if c not in ID_COLS]
N            = len(FEATURE_COLS)

print(f"  {len(df):,} rows, {N} features, {df.province.nunique()} provinces")

TARGET = "dI_dt"
assert TARGET in FEATURE_COLS, f"{TARGET} not found in features"

# Drop rows where dI_dt is NaN
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
print(f"  After dropping dI_dt NaN rows: {len(df):,} rows")

OTHER_FEATURES = [f for f in FEATURE_COLS if f != TARGET]

# ═══════════════════════════════════════════════════════════════
# 1. PEARSON AND SPEARMAN CORRELATIONS AT WEEKLY LEVEL
# ═══════════════════════════════════════════════════════════════
print("\n1. Computing Pearson and Spearman correlations with dI/dt ...")

results = []
for feat in OTHER_FEATURES:
    col = df[feat].fillna(df[feat].mean())
    target = df[TARGET]

    # Drop any remaining NaN in this pair
    mask = col.notna() & target.notna()
    x, y = col[mask].values, target[mask].values

    if len(x) < 30:
        continue

    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)

    results.append({
        "feature":  feat,
        "pearson_r": round(pr, 4),
        "pearson_p": round(pp, 6),
        "spearman_r": round(sr, 4),
        "spearman_p": round(sp, 6),
        "abs_pearson": abs(pr),
    })

res_df = pd.DataFrame(results).sort_values("abs_pearson", ascending=False)
print("\n  TOP 15 features correlated with dI/dt  (Pearson r, weekly data)")
print(f"  {'Feature':<35} {'Pearson r':>10}  {'Spearman r':>10}  {'p-value':>12}")
print("  " + "-" * 72)
for _, row in res_df.head(15).iterrows():
    sig = "***" if row.pearson_p < 0.001 else ("**" if row.pearson_p < 0.01 else "*")
    print(f"  {row.feature:<35} {row.pearson_r:>10.4f}  {row.spearman_r:>10.4f}  {row.pearson_p:>12.2e} {sig}")

res_df.to_csv(os.path.join(OUT_DIR, "didt_correlations.csv"), index=False)
print(f"\n  Full table saved to didt_results/didt_correlations.csv")

# ═══════════════════════════════════════════════════════════════
# 2. LAG ANALYSIS: does feature at t-k predict dI/dt at t?
# ═══════════════════════════════════════════════════════════════
print("\n2. Lag correlation analysis (lags 0 to 4 weeks) ...")

TOP_FEATS_FOR_LAG = res_df.head(10)["feature"].tolist()
LAGS = [0, 1, 2, 3, 4]

lag_matrix = np.full((len(TOP_FEATS_FOR_LAG), len(LAGS)), np.nan)

for fi, feat in enumerate(TOP_FEATS_FOR_LAG):
    for li, lag in enumerate(LAGS):
        rows = []
        for prov, grp in df.groupby("province"):
            grp = grp.sort_values("week")
            x_lagged = grp[feat].shift(lag).values
            y_target = grp[TARGET].values
            mask = ~np.isnan(x_lagged) & ~np.isnan(y_target)
            if mask.sum() < 10:
                continue
            rows.append((x_lagged[mask], y_target[mask]))

        if not rows:
            continue
        all_x = np.concatenate([r[0] for r in rows])
        all_y = np.concatenate([r[1] for r in rows])
        pr, _ = pearsonr(all_x, all_y)
        lag_matrix[fi, li] = pr

print("\n  Lag correlation table  (Pearson r at each lag)")
header = f"  {'Feature':<35}" + "".join(f"  lag{l}" for l in LAGS)
print(header)
print("  " + "-" * 60)
for fi, feat in enumerate(TOP_FEATS_FOR_LAG):
    vals = "".join(f"  {lag_matrix[fi, li]:+.3f}" for li in range(len(LAGS)))
    print(f"  {feat:<35}{vals}")

# ═══════════════════════════════════════════════════════════════
# 3. PER-YEAR BREAKDOWN: how does the correlation change year to year?
# ═══════════════════════════════════════════════════════════════
print("\n3. Per-year correlation breakdown ...")

years = sorted(df["year"].unique())
TOP5  = res_df.head(5)["feature"].tolist()

year_corr = {feat: [] for feat in TOP5}
for yr in years:
    sub = df[df["year"] == yr]
    for feat in TOP5:
        col = sub[feat].fillna(sub[feat].mean())
        tgt = sub[TARGET]
        mask = col.notna() & tgt.notna()
        if mask.sum() < 20:
            year_corr[feat].append(np.nan)
            continue
        pr, _ = pearsonr(col[mask].values, tgt[mask].values)
        year_corr[feat].append(round(pr, 4))

year_corr_df = pd.DataFrame(year_corr, index=years)
year_corr_df.index.name = "year"
print("\n  Per-year Pearson r with dI/dt  (top 5 features)")
print(year_corr_df.to_string())

# ═══════════════════════════════════════════════════════════════
# 4. PROVINCE-LEVEL MEAN |dI/dt|: which provinces are most dynamic?
# ═══════════════════════════════════════════════════════════════
print("\n4. Province-level mean |dI/dt| ...")

prov_dyn = df.groupby("province")[TARGET].apply(lambda x: x.abs().mean()).sort_values(ascending=False)
print("\n  Top 10 most dynamic provinces (highest mean |dI/dt|):")
for prov, val in prov_dyn.head(10).items():
    print(f"    {prov:<35} mean |dI/dt| = {val:.5f}")

# ═══════════════════════════════════════════════════════════════
# 5. PLOT A: Bar chart, Pearson r for all features vs dI/dt
# ═══════════════════════════════════════════════════════════════
print("\n5. Plotting ...")

fig = plt.figure(figsize=(14, 7), facecolor=DARK_BG)
ax  = fig.add_subplot(111)
ax.set_facecolor(PANEL_BG)

colors = [ACC1 if r > 0 else ACC2 for r in res_df["pearson_r"]]
bars = ax.barh(res_df["feature"], res_df["pearson_r"], color=colors, alpha=0.85)

ax.axvline(0, color=TEXT_COL, linewidth=0.8, alpha=0.5)
ax.axvline(0.3,  color=ACC3, linewidth=0.6, linestyle="--", alpha=0.5, label="r=0.3")
ax.axvline(-0.3, color=ACC3, linewidth=0.6, linestyle="--", alpha=0.5)
ax.set_xlabel("Pearson r with dI/dt", color=TEXT_COL, fontsize=10)
ax.set_title("Feature Correlations with dI/dt  (weekly level)",
             color=TEXT_COL, fontsize=13, pad=10)
ax.tick_params(colors=TEXT_COL, labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COL)
ax.grid(True, axis="x", color=GRID_COL, linewidth=0.5, alpha=0.7)
ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "A_didt_bar_correlations.png"),
            dpi=150, facecolor=DARK_BG)
plt.close()

# ═══════════════════════════════════════════════════════════════
# 6. PLOT B: Scatter plots for top 6 features vs dI/dt
# ═══════════════════════════════════════════════════════════════
TOP6 = res_df.head(6)["feature"].tolist()

fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

sample = df.sample(min(3000, len(df)), random_state=42)

for idx, feat in enumerate(TOP6):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    ax.set_facecolor(PANEL_BG)

    x = sample[feat].fillna(sample[feat].mean()).values
    y = sample[TARGET].values
    mask = ~np.isnan(x) & ~np.isnan(y)

    pr = res_df[res_df.feature == feat]["pearson_r"].values[0]
    sr = res_df[res_df.feature == feat]["spearman_r"].values[0]

    ax.scatter(x[mask], y[mask], s=4, alpha=0.35, color=ACC2)

    # Trend line
    if mask.sum() > 10:
        z = np.polyfit(x[mask], y[mask], 1)
        p = np.poly1d(z)
        xline = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xline, p(xline), color=ACC1, linewidth=1.5, alpha=0.9)

    ax.set_xlabel(feat, color=TEXT_COL, fontsize=8)
    ax.set_ylabel("dI/dt", color=TEXT_COL, fontsize=8)
    ax.set_title(f"r={pr:+.3f}  sr={sr:+.3f}", color=TEXT_COL, fontsize=9, pad=4)
    ax.tick_params(colors=TEXT_COL, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.4, alpha=0.5)

fig.suptitle("Top 6 Feature Correlates of dI/dt  (3000-sample scatter)",
             color=TEXT_COL, fontsize=13, y=1.01)
plt.savefig(os.path.join(OUT_DIR, "B_didt_scatter_top6.png"),
            dpi=150, facecolor=DARK_BG, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════
# 7. PLOT C: Lag correlation heatmap
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(8, 5), facecolor=DARK_BG)
ax  = fig.add_subplot(111)
ax.set_facecolor(PANEL_BG)

norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
im   = ax.imshow(lag_matrix, cmap="RdBu_r", norm=norm, aspect="auto")

ax.set_xticks(range(len(LAGS)))
ax.set_xticklabels([f"lag {l}w" for l in LAGS], color=TEXT_COL, fontsize=9)
ax.set_yticks(range(len(TOP_FEATS_FOR_LAG)))
ax.set_yticklabels(TOP_FEATS_FOR_LAG, color=TEXT_COL, fontsize=8)

for i in range(len(TOP_FEATS_FOR_LAG)):
    for j in range(len(LAGS)):
        v = lag_matrix[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(v) > 0.3 else "#aaaacc")

cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cb.ax.tick_params(colors=TEXT_COL, labelsize=8)
cb.set_label("Pearson r", color=TEXT_COL, fontsize=9)

ax.set_title("Lag Correlation: Feature[t-k] vs dI/dt[t]  (top 10 features)",
             color=TEXT_COL, fontsize=11, pad=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "C_didt_lag_heatmap.png"),
            dpi=150, facecolor=DARK_BG)
plt.close()

# ═══════════════════════════════════════════════════════════════
# 8. PLOT D: Per-year correlation line chart for top 5 features
# ═══════════════════════════════════════════════════════════════
PALETTE = [ACC1, ACC2, ACC3, "#a786c9", "#06d6a0"]

fig = plt.figure(figsize=(11, 5), facecolor=DARK_BG)
ax  = fig.add_subplot(111)
ax.set_facecolor(PANEL_BG)

for fi, feat in enumerate(TOP5):
    vals = year_corr_df[feat].values
    ax.plot(years, vals, color=PALETTE[fi], linewidth=2,
            marker="o", markersize=5, label=feat)

ax.axhline(0, color=TEXT_COL, linewidth=0.7, alpha=0.5, linestyle="--")
ax.set_xlabel("Year", color=TEXT_COL, fontsize=10)
ax.set_ylabel("Pearson r with dI/dt", color=TEXT_COL, fontsize=10)
ax.set_title("How Feature vs dI/dt Correlation Shifts Year by Year",
             color=TEXT_COL, fontsize=12, pad=8)
ax.tick_params(colors=TEXT_COL, labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COL)
ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.6)
ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COL,
          loc="upper left", framealpha=0.8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "D_didt_year_breakdown.png"),
            dpi=150, facecolor=DARK_BG)
plt.close()

# ═══════════════════════════════════════════════════════════════
# 9. WRITTEN SUMMARY
# ═══════════════════════════════════════════════════════════════
top1 = res_df.iloc[0]
top3 = res_df.head(3)

with open(os.path.join(OUT_DIR, "didt_summary.txt"), "w") as f:
    f.write("dI/dt CORRELATION ANALYSIS\n")
    f.write("=" * 60 + "\n\n")

    f.write("WHAT IS dI/dt\n")
    f.write("-" * 40 + "\n")
    f.write("dI/dt is the weekly rate of change of dengue incidence.\n")
    f.write("It is the quantity SINDy will learn to model.\n")
    f.write("Understanding what drives it tells us which terms\n")
    f.write("should appear in the governing equation library.\n\n")

    f.write("TOP FEATURES BY PEARSON r (weekly level)\n")
    f.write("-" * 40 + "\n")
    for _, row in res_df.head(10).iterrows():
        direction = "positive" if row.pearson_r > 0 else "negative"
        f.write(f"  {row.feature:<35} r={row.pearson_r:+.4f} ({direction})\n")

    f.write("\nKEY FINDINGS\n")
    f.write("-" * 40 + "\n")
    f.write(f"1. Strongest correlate is {top1.feature} at r={top1.pearson_r:+.4f}.\n")
    f.write("   This should be a primary term in the SINDy library.\n\n")

    pos_feats = res_df[res_df.pearson_r > 0.2].feature.tolist()
    neg_feats = res_df[res_df.pearson_r < -0.2].feature.tolist()

    f.write(f"2. Features with positive correlation (r > 0.2): {len(pos_feats)}\n")
    for feat in pos_feats:
        f.write(f"   {feat}\n")

    f.write(f"\n3. Features with negative correlation (r < -0.2): {len(neg_feats)}\n")
    for feat in neg_feats:
        f.write(f"   {feat}\n")

    f.write("\nLAG ANALYSIS SUMMARY\n")
    f.write("-" * 40 + "\n")
    f.write("Lag 0: contemporaneous effect (same week)\n")
    f.write("Lag 1: does last week predict this week dI/dt?\n")
    f.write("If lag 0 is stronger than lag 1, the relationship is\n")
    f.write("concurrent, not predictive. If lag 1 is similar to lag 0,\n")
    f.write("there is genuine predictive signal worth including in SINDy.\n\n")

    f.write("MOST DYNAMIC PROVINCES (highest mean |dI/dt|)\n")
    f.write("-" * 40 + "\n")
    for prov, val in prov_dyn.head(10).items():
        f.write(f"  {prov:<35} mean |dI/dt| = {val:.5f}\n")

    f.write("\nOUTPUTS\n")
    f.write("-" * 40 + "\n")
    f.write("  didt_correlations.csv       full ranked table\n")
    f.write("  A_didt_bar_correlations.png bar chart, all features\n")
    f.write("  B_didt_scatter_top6.png     scatter top 6 vs dI/dt\n")
    f.write("  C_didt_lag_heatmap.png      lag 0-4 week heatmap\n")
    f.write("  D_didt_year_breakdown.png   yearly stability of correlations\n")

print("\nDone. Outputs saved to analysis/didt_results/")
print("  A_didt_bar_correlations.png")
print("  B_didt_scatter_top6.png")
print("  C_didt_lag_heatmap.png")
print("  D_didt_year_breakdown.png")
print("  didt_correlations.csv")
print("  didt_summary.txt")
