"""
plot_r2_distribution.py
────────────────────────────────────────────────────────────────────────────
Visualise the distribution of SINDy M1 R² scores across all 32 DR provinces
for the validation (2022) and test (2023) cohorts.

Produces:  sindy/results/r2_distribution.png
────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ── Paths ──────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
M1_SUMMARY  = os.path.join(HERE, "results", "m1_summary.csv")
OUT_PATH    = os.path.join(HERE, "results", "r2_distribution.png")

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(M1_SUMMARY)
df = df.sort_values("r2_test", ascending=False).reset_index(drop=True)

val  = df["r2_val"].values
test = df["r2_test"].values
prov = df["province"].str.replace(r"^\d+ ", "", regex=True).values   # strip number prefix

n = len(df)

# ── Colour helpers ─────────────────────────────────────────────────────────
VAL_COLOR  = "#1976D2"   # blue
TEST_COLOR = "#D32F2F"   # red
GRID_COLOR = "#e8e8e8"

def bar_alpha(val):
    """Slightly dim negative-R² bars to visually flag them."""
    return 0.85 if val >= 0 else 0.55

# ══════════════════════════════════════════════════════════════════════════
# Figure layout:  3 panels stacked vertically
#   Panel A  – grouped bar chart (val vs test) per province, sorted by test R²
#   Panel B  – KDE + histogram overlay for val and test distributions
#   Panel C  – cumulative distribution (ECDF) for val and test
# ══════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 15))
fig.patch.set_facecolor("white")

gs = fig.add_gridspec(
    3, 1,
    height_ratios=[2.2, 1.4, 1.4],
    hspace=0.42,
    left=0.07, right=0.97,
    top=0.93, bottom=0.06,
)

# ── Panel A: Grouped bar chart ─────────────────────────────────────────────
ax_bar = fig.add_subplot(gs[0])

x      = np.arange(n)
width  = 0.38

# Draw val bars
for i, (xi, v) in enumerate(zip(x, val)):
    ax_bar.bar(xi - width/2, v, width,
               color=VAL_COLOR, alpha=bar_alpha(v), zorder=3)

# Draw test bars
for i, (xi, t) in enumerate(zip(x, test)):
    ax_bar.bar(xi + width/2, t, width,
               color=TEST_COLOR, alpha=bar_alpha(t), zorder=3)

# Reference line at R²=0
ax_bar.axhline(0,  color="#555", linewidth=0.8, linestyle="--", zorder=2)
# Reference line at SINDy mean test R²
mean_test = float(np.mean(test))
mean_val  = float(np.mean(val))
ax_bar.axhline(mean_test, color=TEST_COLOR, linewidth=1.2,
               linestyle=":", alpha=0.7, zorder=2)
ax_bar.axhline(mean_val, color=VAL_COLOR, linewidth=1.2,
               linestyle=":", alpha=0.7, zorder=2)

# Shade the negative zone
ax_bar.axhspan(-1, 0, color="#ffebee", alpha=0.45, zorder=1)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(prov, rotation=45, ha="right", fontsize=7.5)
ax_bar.set_ylabel("R²", fontsize=11)
ax_bar.set_title("A   Per-province R² — Validation (2022) vs Test (2023)",
                 fontsize=12, fontweight="bold", loc="left", pad=8)
ax_bar.set_xlim(-0.7, n - 0.3)
ax_bar.set_ylim(-0.35, 1.02)
ax_bar.yaxis.set_minor_locator(MultipleLocator(0.1))
ax_bar.grid(axis="y", color=GRID_COLOR, zorder=0)
ax_bar.set_facecolor("white")

# Annotate mean lines
ax_bar.text(n - 0.5, mean_val + 0.02, f"μ_val={mean_val:.3f}",
            color=VAL_COLOR, fontsize=8, ha="right")
ax_bar.text(n - 0.5, mean_test - 0.05, f"μ_test={mean_test:.3f}",
            color=TEST_COLOR, fontsize=8, ha="right")

# Flag the two negative-test provinces
for i, (xi, t, name) in enumerate(zip(x, test, prov)):
    if t < 0:
        ax_bar.annotate(
            f"{t:.2f}", xy=(xi + width/2, t - 0.02),
            ha="center", va="top", fontsize=7, color=TEST_COLOR, fontweight="bold"
        )

patch_val  = mpatches.Patch(color=VAL_COLOR, alpha=0.85, label=f"Validation 2022 (n={n})")
patch_test = mpatches.Patch(color=TEST_COLOR, alpha=0.85, label=f"Test 2023 (n={n})")
ax_bar.legend(handles=[patch_val, patch_test], fontsize=9,
              loc="upper right", framealpha=0.9)


# ── Panel B: Histogram + KDE ───────────────────────────────────────────────
ax_hist = fig.add_subplot(gs[1])

bins = np.linspace(-0.25, 1.05, 22)

ax_hist.hist(val,  bins=bins, color=VAL_COLOR,  alpha=0.55, label="Val 2022",  zorder=3)
ax_hist.hist(test, bins=bins, color=TEST_COLOR, alpha=0.55, label="Test 2023", zorder=3)

# Simple KDE via Gaussian smoothing
from scipy.stats import gaussian_kde
kde_x = np.linspace(-0.35, 1.1, 300)
for data, color in [(val, VAL_COLOR), (test, TEST_COLOR)]:
    kde = gaussian_kde(data, bw_method=0.35)
    # scale to approximate histogram height
    scale = len(data) * (bins[1] - bins[0])
    ax_hist.plot(kde_x, kde(kde_x) * scale, color=color, linewidth=2.2, zorder=4)

ax_hist.axvline(0, color="#555", linewidth=0.8, linestyle="--", zorder=2)
ax_hist.axvspan(-0.35, 0, color="#ffebee", alpha=0.4, zorder=1)

ax_hist.set_xlabel("R²", fontsize=11)
ax_hist.set_ylabel("Province count", fontsize=11)
ax_hist.set_title("B   Score distribution with KDE",
                  fontsize=12, fontweight="bold", loc="left", pad=8)
ax_hist.legend(fontsize=9, framealpha=0.9)
ax_hist.grid(color=GRID_COLOR, zorder=0)
ax_hist.set_facecolor("white")

# Annotate quantiles
for data, color, label in [(val, VAL_COLOR, "val"), (test, TEST_COLOR, "test")]:
    med = np.median(data)
    ax_hist.axvline(med, color=color, linewidth=1.4, linestyle="--", alpha=0.8, zorder=5)
    ax_hist.text(med + 0.01, ax_hist.get_ylim()[1] * 0.85,
                 f"med_{label}={med:.2f}", color=color, fontsize=8, rotation=90, va="top")


# ── Panel C: ECDF ──────────────────────────────────────────────────────────
ax_ecdf = fig.add_subplot(gs[2])

for data, color, label in [(val, VAL_COLOR, "Val 2022"), (test, TEST_COLOR, "Test 2023")]:
    sorted_d = np.sort(data)
    ecdf     = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax_ecdf.step(sorted_d, ecdf, where="post", color=color, linewidth=2.2,
                 label=label, zorder=3)
    # 25th, 50th, 75th percentile markers
    for q, marker in [(0.25, "v"), (0.50, "s"), (0.75, "^")]:
        xq = np.quantile(data, q)
        ax_ecdf.plot(xq, q, marker=marker, color=color, markersize=6, zorder=5)
        ax_ecdf.axvline(xq, color=color, linewidth=0.6, linestyle=":", alpha=0.5, zorder=2)

ax_ecdf.axvline(0, color="#555", linewidth=0.8, linestyle="--", zorder=2)
ax_ecdf.axvspan(-1, 0, color="#ffebee", alpha=0.4, zorder=1)
ax_ecdf.axhline(0.5, color="#aaa", linewidth=0.7, linestyle="--", zorder=1)

ax_ecdf.set_xlabel("R²", fontsize=11)
ax_ecdf.set_ylabel("Cumulative fraction", fontsize=11)
ax_ecdf.set_title("C   Empirical CDF  (▼=Q1  ■=median  ▲=Q3)",
                  fontsize=12, fontweight="bold", loc="left", pad=8)
ax_ecdf.set_xlim(-0.3, 1.05)
ax_ecdf.set_ylim(0, 1.05)
ax_ecdf.legend(fontsize=9, framealpha=0.9)
ax_ecdf.grid(color=GRID_COLOR, zorder=0)
ax_ecdf.set_facecolor("white")

# ── Stats table (text in figure) ──────────────────────────────────────────
def stats_str(data, label):
    pos = np.sum(data > 0)
    return (f"{label}: mean={np.mean(data):.3f}  median={np.median(data):.3f}  "
            f"std={np.std(data):.3f}  min={np.min(data):.3f}  "
            f"max={np.max(data):.3f}  R²>0: {pos}/{len(data)}")

fig.text(0.5, 0.005,
         stats_str(val, "Val") + "    |    " + stats_str(test, "Test"),
         ha="center", fontsize=8.5, color="#444",
         bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#ddd"))

fig.suptitle("SINDy M1 — R² Score Distribution Across 32 DR Provinces",
             fontsize=14, fontweight="bold", y=0.97)

# ── Save ───────────────────────────────────────────────────────────────────
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT_PATH}")

# ── Print summary table ────────────────────────────────────────────────────
print("\n── Summary statistics ────────────────────────────────")
print(f"{'Metric':<20} {'Val 2022':>12} {'Test 2023':>12}")
print("-" * 46)
for label, data in [("Mean R²", [np.mean(val), np.mean(test)]),
                    ("Median R²",  [np.median(val), np.median(test)]),
                    ("Std dev",    [np.std(val), np.std(test)]),
                    ("Min R²",     [np.min(val), np.min(test)]),
                    ("Max R²",     [np.max(val), np.max(test)]),
                    ("R² > 0",     [np.sum(val > 0), np.sum(test > 0)]),
                    ("R² > 0.5",   [np.sum(val > 0.5), np.sum(test > 0.5)])]:
    print(f"{label:<20} {data[0]:>12.3f} {data[1]:>12.3f}")

print("\n── Provinces with test R² < 0 ────────────────────────")
neg = df[df["r2_test"] < 0][["province", "r2_val", "r2_test", "n_nonzero"]]
print(neg.to_string(index=False))

print("\n── Provinces with train→test gap > 0.4 ──────────────")
df["gap"] = df["r2_train"] - df["r2_test"]
gap = df[df["gap"] > 0.4][["province", "r2_train", "r2_val", "r2_test", "gap", "n_nonzero"]]
print(gap.to_string(index=False))
