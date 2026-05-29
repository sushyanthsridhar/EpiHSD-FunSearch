import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import periodogram
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings, os
warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
INPUT       = "timeseries_all_provinces.csv"
OUTPUT_DIR  = "province_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT)
provinces = sorted(df["province"].unique())
N = len(provinces)

# ── helper: get clean series for a province ───────────────────────────────────
def get_series(prov):
    sub = df[df["province"] == prov].sort_values(["year","week"])
    return sub["total_cases"].values.astype(float)

# ─────────────────────────────────────────────────────────────────────────────
# 1. BASIC STATS
# ─────────────────────────────────────────────────────────────────────────────
print("Computing basic stats...")
rows = []
for prov in provinces:
    s = get_series(prov)
    nz = s[s > 0]
    rows.append({
        "province"      : prov,
        "total_cases"   : int(s.sum()),
        "mean_wk"       : round(s.mean(), 2),
        "median_wk"     : round(np.median(s), 2),
        "std_wk"        : round(s.std(), 2),
        "cv"            : round(s.std() / s.mean() if s.mean() > 0 else 0, 3),
        "peak"          : int(s.max()),
        "zero_pct"      : round((s == 0).mean() * 100, 1),
        "active_weeks"  : int((s > 0).sum()),
    })
stats_df = pd.DataFrame(rows).sort_values("total_cases", ascending=False).reset_index(drop=True)
stats_df.to_csv(f"{OUTPUT_DIR}/01_basic_stats.csv", index=False)
print(f"  -> {OUTPUT_DIR}/01_basic_stats.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 2. AUTOREGRESSIVE ANALYSIS (ACF + partial ACF + AR fit quality)
# ─────────────────────────────────────────────────────────────────────────────
print("Computing autoregressive metrics...")

def acf(x, max_lag=52):
    x = x - x.mean()
    n = len(x)
    denom = np.dot(x, x)
    return [1.0] + [np.dot(x[:n-l], x[l:]) / denom for l in range(1, max_lag+1)]

def ar1_r2(x):
    """Fit AR(1): x[t] = a*x[t-1] + b, return R2 and coefficient"""
    y, X = x[1:], x[:-1]
    slope, intercept, r, p, _ = stats.linregress(X, y)
    return round(r**2, 4), round(slope, 4), round(p, 6)

def ar_memory(acf_vals, threshold=0.2):
    """How many lags until ACF drops below threshold"""
    for i, v in enumerate(acf_vals[1:], 1):
        if abs(v) < threshold:
            return i
    return len(acf_vals)

ar_rows = []
MAX_LAG = 52
acf_matrix = np.zeros((N, MAX_LAG + 1))

for i, prov in enumerate(provinces):
    s = get_series(prov)
    acf_vals = acf(s, MAX_LAG)
    acf_matrix[i] = acf_vals
    r2, coef, pval = ar1_r2(s)
    lag1  = round(acf_vals[1], 4)
    lag4  = round(acf_vals[4], 4)
    lag13 = round(acf_vals[13], 4) if len(acf_vals) > 13 else 0
    lag52 = round(acf_vals[52], 4) if len(acf_vals) > 52 else 0
    mem   = ar_memory(acf_vals)
    ar_rows.append({
        "province"      : prov,
        "AR1_coef"      : coef,
        "AR1_R2"        : r2,
        "AR1_pvalue"    : pval,
        "ACF_lag1"      : lag1,
        "ACF_lag4"      : lag4,
        "ACF_lag13"     : lag13,
        "ACF_lag52"     : lag52,
        "AR_memory_wks" : mem,
        "AR_useful"     : "YES" if r2 > 0.15 and pval < 0.05 else "WEAK",
    })

ar_df = pd.DataFrame(ar_rows).sort_values("AR1_R2", ascending=False).reset_index(drop=True)
ar_df.to_csv(f"{OUTPUT_DIR}/02_autoregressive.csv", index=False)
print(f"  -> {OUTPUT_DIR}/02_autoregressive.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SEASONALITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("Computing seasonality...")

def dominant_period(x):
    """Return dominant frequency period (in weeks) from periodogram"""
    f, pxx = periodogram(x - x.mean())
    f, pxx = f[1:], pxx[1:]  # drop DC
    if pxx.max() == 0:
        return None, 0
    dom_f = f[np.argmax(pxx)]
    return round(1/dom_f, 1) if dom_f > 0 else None, round(pxx.max(), 2)

def monthly_profile(prov):
    sub = df[df["province"]==prov].copy()
    sub = sub[sub["month"] > 0]
    monthly = sub.groupby("month")["total_cases"].mean()
    peak_month = int(monthly.idxmax()) if len(monthly) > 0 else 0
    trough_month = int(monthly.idxmin()) if len(monthly) > 0 else 0
    seasonal_ratio = round(monthly.max() / monthly.mean(), 3) if monthly.mean() > 0 else 0
    return peak_month, trough_month, seasonal_ratio

MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec",0:"?"}

sea_rows = []
for prov in provinces:
    s = get_series(prov)
    dom_period, dom_power = dominant_period(s)
    peak_m, trough_m, seas_ratio = monthly_profile(prov)
    sea_rows.append({
        "province"         : prov,
        "dominant_period_wks": dom_period,
        "spectral_power"   : dom_power,
        "peak_month"       : MONTH_NAMES.get(peak_m, "?"),
        "trough_month"     : MONTH_NAMES.get(trough_m, "?"),
        "seasonal_ratio"   : seas_ratio,
        "strong_seasonality": "YES" if seas_ratio > 1.5 else "WEAK",
    })

sea_df = pd.DataFrame(sea_rows).sort_values("seasonal_ratio", ascending=False).reset_index(drop=True)
sea_df.to_csv(f"{OUTPUT_DIR}/03_seasonality.csv", index=False)
print(f"  -> {OUTPUT_DIR}/03_seasonality.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PROVINCE SIMILARITY (cosine similarity on normalized series)
# ─────────────────────────────────────────────────────────────────────────────
print("Computing province similarity...")

series_matrix = np.array([get_series(p) for p in provinces])
scaler = StandardScaler()
series_norm = scaler.fit_transform(series_matrix.T).T
sim_matrix = cosine_similarity(series_norm)
sim_df = pd.DataFrame(sim_matrix, index=provinces, columns=provinces)
sim_df.to_csv(f"{OUTPUT_DIR}/04_province_similarity.csv")

# Find top 3 most similar pairs
sim_rows = []
for i in range(N):
    for j in range(i+1, N):
        sim_rows.append({"prov_A": provinces[i], "prov_B": provinces[j], "similarity": round(sim_matrix[i,j], 4)})
sim_pairs = pd.DataFrame(sim_rows).sort_values("similarity", ascending=False)
sim_pairs.to_csv(f"{OUTPUT_DIR}/04b_similar_pairs.csv", index=False)
print(f"  -> {OUTPUT_DIR}/04_province_similarity.csv")
print(f"  -> {OUTPUT_DIR}/04b_similar_pairs.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SIGNAL QUALITY / SHAPE SCORE (for SINDy/Funsearch suitability)
# ─────────────────────────────────────────────────────────────────────────────
print("Computing shape/quality scores...")

def shape_score(prov):
    s = get_series(prov)
    # Components
    volume    = min(s.sum() / 5000, 1.0)          # normalized by 5000 cases
    density   = (s > 0).mean()                     # fraction of active weeks
    snr       = s.mean() / (s.std() + 1e-6)       # signal to noise
    snr_score = min(snr / 0.5, 1.0)
    ar1_r2, _, _ = ar1_r2_fn(s)
    # Penalize spikiness
    spike_pen = 1 - min((s.max() / (s.mean() + 1e-6)) / 50, 1.0)
    # Composite
    score = (0.30 * volume + 0.25 * density + 0.20 * snr_score +
             0.15 * ar1_r2 + 0.10 * spike_pen)
    return round(score * 100, 1)

def ar1_r2_fn(x):
    y, X = x[1:], x[:-1]
    slope, intercept, r, p, _ = stats.linregress(X, y)
    return round(r**2, 4), round(slope, 4), round(p, 6)

qual_rows = []
for prov in provinces:
    s = get_series(prov)
    score = shape_score(prov)
    r2, _, _ = ar1_r2_fn(s)
    zero_pct = (s == 0).mean() * 100
    cv = s.std() / (s.mean() + 1e-6)
    tier = ("T1-Excellent" if score >= 55 else
            "T2-Good"      if score >= 35 else
            "T3-Marginal"  if score >= 20 else
            "T4-Skip")
    qual_rows.append({
        "province"   : prov,
        "shape_score": score,
        "tier"       : tier,
        "AR1_R2"     : r2,
        "zero_pct"   : round(zero_pct, 1),
        "cv"         : round(cv, 3),
        "total_cases": int(s.sum()),
    })

qual_df = pd.DataFrame(qual_rows).sort_values("shape_score", ascending=False).reset_index(drop=True)
qual_df.to_csv(f"{OUTPUT_DIR}/05_shape_quality.csv", index=False)
print(f"  -> {OUTPUT_DIR}/05_shape_quality.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE UTILITY SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("Computing feature utility...")

feat_rows = []
for prov in provinces:
    s = get_series(prov)
    sub = df[df["province"]==prov].copy()

    # Lag features: how much do lag1..lag8 explain variance?
    lag_r2 = []
    for lag in range(1, 9):
        y = s[lag:]; X = s[:-lag]
        _, _, r, _, _ = stats.linregress(X, y)
        lag_r2.append(round(r**2, 3))

    # Month as feature: does monthly grouping explain variance?
    sub2 = sub[sub["month"] > 0]
    monthly_means = sub2.groupby("month")["total_cases"].mean()
    month_mapped  = sub2["month"].map(monthly_means)
    if len(month_mapped) > 1:
        _, _, r_month, _, _ = stats.linregress(month_mapped, sub2["total_cases"])
        month_r2 = round(r_month**2, 3)
    else:
        month_r2 = 0

    # Year trend: is there a long-term trend?
    yearly = sub.groupby("year")["total_cases"].sum().values.astype(float)
    if len(yearly) > 2:
        _, _, r_yr, _, _ = stats.linregress(range(len(yearly)), yearly)
        year_r2 = round(r_yr**2, 3)
    else:
        year_r2 = 0

    feat_rows.append({
        "province"    : prov,
        "lag1_R2"     : lag_r2[0],
        "lag2_R2"     : lag_r2[1],
        "lag4_R2"     : lag_r2[3],
        "lag8_R2"     : lag_r2[7],
        "month_R2"    : month_r2,
        "year_trend_R2": year_r2,
        "best_lag"    : int(np.argmax(lag_r2) + 1),
        "lag_useful"  : "YES" if max(lag_r2) > 0.15 else "WEAK",
        "month_useful": "YES" if month_r2 > 0.05 else "WEAK",
        "trend_useful": "YES" if year_r2 > 0.1 else "WEAK",
    })

feat_df = pd.DataFrame(feat_rows).sort_values("lag1_R2", ascending=False).reset_index(drop=True)
feat_df.to_csv(f"{OUTPUT_DIR}/06_feature_utility.csv", index=False)
print(f"  -> {OUTPUT_DIR}/06_feature_utility.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("Generating plots...")

# 7a. Shape score ranking
fig, ax = plt.subplots(figsize=(12, 8))
colors = {"T1-Excellent":"#1D9E75","T2-Good":"#378ADD","T3-Marginal":"#EF9F27","T4-Skip":"#E24B4A"}
q = qual_df.sort_values("shape_score")
bars = ax.barh(q["province"], q["shape_score"],
               color=[colors[t] for t in q["tier"]], height=0.7)
ax.set_xlabel("Shape score (0-100)", fontsize=11)
ax.set_title("Province shape/quality score for SINDy & Funsearch", fontsize=13, fontweight="bold")
for tier, col in colors.items():
    ax.barh([], [], color=col, label=tier)
ax.legend(fontsize=9)
ax.axvline(55, color="#1D9E75", linestyle="--", alpha=0.5, linewidth=1)
ax.axvline(35, color="#378ADD", linestyle="--", alpha=0.5, linewidth=1)
ax.axvline(20, color="#EF9F27", linestyle="--", alpha=0.5, linewidth=1)
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_shape_scores.png", dpi=150)
plt.close()
print(f"  -> {OUTPUT_DIR}/plot_shape_scores.png")

# 7b. ACF heatmap across all provinces
fig, ax = plt.subplots(figsize=(14, 9))
im = ax.imshow(acf_matrix[:, 1:27], aspect="auto", cmap="RdYlGn",
               vmin=-0.4, vmax=0.8, interpolation="nearest")
ax.set_yticks(range(N))
ax.set_yticklabels(provinces, fontsize=7)
ax.set_xticks(range(0, 26, 4))
ax.set_xticklabels([f"lag {i+1}" for i in range(0, 26, 4)], fontsize=8)
ax.set_title("ACF heatmap — all provinces (lags 1-26)", fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax, label="Autocorrelation")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_acf_heatmap.png", dpi=150)
plt.close()
print(f"  -> {OUTPUT_DIR}/plot_acf_heatmap.png")

# 7c. Similarity heatmap
fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(sim_matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(N)); ax.set_yticks(range(N))
ax.set_xticklabels(provinces, rotation=90, fontsize=6)
ax.set_yticklabels(provinces, fontsize=6)
ax.set_title("Province similarity matrix (cosine similarity on normalized series)", fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_similarity_matrix.png", dpi=150)
plt.close()
print(f"  -> {OUTPUT_DIR}/plot_similarity_matrix.png")

# 7d. Feature utility bubble chart
fig, ax = plt.subplots(figsize=(12, 7))
ax.scatter(feat_df["lag1_R2"], feat_df["month_R2"],
           s=feat_df["lag1_R2"] * 800 + 30,
           c=qual_df.set_index("province").loc[feat_df["province"], "shape_score"].values,
           cmap="RdYlGn", alpha=0.8, edgecolors="white", linewidth=0.5)
for _, row in feat_df.iterrows():
    ax.annotate(row["province"].split(" ",1)[0], (row["lag1_R2"], row["month_R2"]),
                fontsize=6.5, ha="center", va="bottom")
ax.set_xlabel("Lag-1 R² (autoregressive signal)", fontsize=11)
ax.set_ylabel("Month R² (seasonal signal)", fontsize=11)
ax.set_title("Feature utility: AR vs seasonality — bubble size = AR strength, color = shape score", fontsize=11, fontweight="bold")
ax.grid(linestyle="--", alpha=0.4)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_feature_utility.png", dpi=150)
plt.close()
print(f"  -> {OUTPUT_DIR}/plot_feature_utility.png")

# 7e. Top 5 province time series
top5 = qual_df.head(5)["province"].tolist()
fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
x = range(468)
xtick_pos  = [i*52 for i in range(9)]
xtick_lbls = list(range(2015, 2024))
for ax, prov in zip(axes, top5):
    s = get_series(prov)
    ax.plot(x, s, linewidth=1.2, color="#378ADD")
    ax.fill_between(x, s, alpha=0.15, color="#378ADD")
    ax.set_ylabel("Cases/wk", fontsize=8)
    ax.set_title(prov, fontsize=9, fontweight="bold", loc="left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)
    for xp in xtick_pos:
        ax.axvline(xp, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
axes[-1].set_xticks(xtick_pos)
axes[-1].set_xticklabels(xtick_lbls)
fig.suptitle("Top 5 provinces by shape score — weekly time series 2015-2023",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot_top5_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {OUTPUT_DIR}/plot_top5_timeseries.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. MASTER SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("Building master summary...")
master = (qual_df[["province","shape_score","tier","total_cases","zero_pct","cv"]]
          .merge(ar_df[["province","AR1_R2","AR1_coef","AR_memory_wks","AR_useful"]], on="province")
          .merge(sea_df[["province","dominant_period_wks","peak_month","seasonal_ratio","strong_seasonality"]], on="province")
          .merge(feat_df[["province","lag1_R2","month_R2","year_trend_R2","best_lag","lag_useful","month_useful"]], on="province")
          .sort_values("shape_score", ascending=False)
          .reset_index(drop=True))
master.to_csv(f"{OUTPUT_DIR}/00_MASTER_SUMMARY.csv", index=False)
print(f"  -> {OUTPUT_DIR}/00_MASTER_SUMMARY.csv")

print(f"""
╔══════════════════════════════════════════════════════╗
║  All outputs saved to ./{OUTPUT_DIR}/
║
║  CSVs:
║    00_MASTER_SUMMARY.csv      ← start here
║    01_basic_stats.csv
║    02_autoregressive.csv
║    03_seasonality.csv
║    04_province_similarity.csv
║    04b_similar_pairs.csv
║    05_shape_quality.csv
║    06_feature_utility.csv
║
║  Plots:
║    plot_shape_scores.png
║    plot_acf_heatmap.png
║    plot_similarity_matrix.png
║    plot_feature_utility.png
║    plot_top5_timeseries.png
╚══════════════════════════════════════════════════════╝
""")