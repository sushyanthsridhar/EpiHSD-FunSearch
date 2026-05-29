"""
run_sindy.py
────────────────────────────────────────────────────────────────────────────
SINDy equation discovery for dengue fever — Dominican Republic.
Target variable: dI_dt_norm  (weekly change in incidence per 100,000 people)

Reads:   sindy/seir_features.csv   (built by build_seir_features.py)
Writes:  sindy/results/
           m1_per_province/        — Method 1 fit plots + per-province CSV
           m1_summary.csv          — weighted R² table (Method 1)
           m2_fit_<province>.png   — Method 2 test-province plots
           m2_summary.csv          — Method 2 global equation + R²s

════════════════════════════════════════════════════════════════════════════
METHOD 1 — Province-by-province, temporal split, shape-weighted R²
════════════════════════════════════════════════════════════════════════════
  For each of the 32 provinces (sorted top→low by total cases):
    • Temporal split within that province:
        TRAIN  2015–2021   (7 years, ~360 weeks)
        VAL    2022        (52 weeks)  — grid search picks threshold here
        TEST   2023        (52 weeks)  — R² reported here
    • SINDy fitted on TRAIN, threshold chosen on VAL, evaluated on TEST.
    • shape_score  = max(weekly_incidence) / mean(weekly_incidence)
                   High peaks relative to baseline → more epidemic signal
                   → higher weight in the weighted average R².
    • Weighted R²  = Σ(shape_score_i × R²_test_i) / Σ(shape_score_i)

  Why: Tests whether SINDy can discover a governing equation for each
  province that forecasts 2023 from 2015–2021 history.  The weighted
  average rewards provinces where the epidemic signal is cleanest.

════════════════════════════════════════════════════════════════════════════
METHOD 2 — All provinces pooled, temporal split
════════════════════════════════════════════════════════════════════════════
  All 32 provinces pooled into one dataset, split by year:
    TRAIN  2015–2021   (~11,776 rows across all provinces)
    VAL    2022        (~1,664 rows)
    TEST   2023        (~1,664 rows)
  One universal equation fitted.  Test R² reported per province + pooled.

  Why: Tests whether a single universal law holds across the entire DR.

Usage:
  python sindy/run_sindy.py
────────────────────────────────────────────────────────────────────────────
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pysindy as ps
except ImportError:
    raise ImportError("pysindy not installed.  Run:  pip install pysindy")

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE, "seir_features.csv")
OUT_DIR   = os.path.join(BASE, "results")
M1_DIR    = os.path.join(OUT_DIR, "m1_per_province")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(M1_DIR,   exist_ok=True)

# ── Temporal split years ──────────────────────────────────────────────────────
TRAIN_YEARS = list(range(2015, 2022))   # 2015–2021
VAL_YEARS   = [2022]
TEST_YEARS  = [2023]

# ── SINDy settings ────────────────────────────────────────────────────────────
POLY_DEGREE      = 2
GRID_SEARCH      = True
R2_TOLERANCE     = 0.10
THRESHOLD_VALUES = list(np.round(np.linspace(0.01, 2.0, 80), 4))
BEST_THRESHOLD   = 0.30     # fallback if GRID_SEARCH = False

# ── Features (normalised per 100k throughout) ─────────────────────────────────
# Target  : dI_dt_norm  (weekly Δincidence per 100k)
# Features: incidence_lag1 is col 0 → its derivative = dI_dt_norm
FEATURE_COLS = [
    "incidence_lag1",        # I(t-1) per 100k  [index 0, SINDy target]
    "EIP",                   # exp(0.347·T − 9.17)  thermal transmission forcing
    "temp_lag4",             # temperature 4 weeks prior
    "rainfall_lag3",         # rainfall 3 weeks prior
    "avg_humidity",          # weekly mean relative humidity
    "incidence_lag2",        # I(t-2) per 100k
    "incidence_lag4",        # I(t-4) per 100k
    "rolling_incidence_4wk", # 4-week rolling mean incidence
    "week_sin",              # seasonal forcing
    "week_cos",
]
TARGET_COL = "dI_dt_norm"   # normalised target (per 100k per week)
TARGET_IDX = 0              # incidence_lag1 is column 0

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading seir_features.csv …")
df = pd.read_csv(INPUT_CSV)
print(f"  Shape: {df.shape}  |  Provinces: {df['province'].nunique()}")

# Province ranking (top → low by total cases)
province_totals = (df.groupby("province")["total_cases"]
                     .sum().sort_values(ascending=False))
all_provinces = province_totals.index.tolist()


# ── Shared helpers ────────────────────────────────────────────────────────────
def make_Xy(data: pd.DataFrame):
    X     = data[FEATURE_COLS].values.astype(float)
    y     = data[TARGET_COL].values.astype(float)
    X_dot = np.zeros_like(X)
    X_dot[:, TARGET_IDX] = y
    return X, X_dot, y


def build_and_fit(X, X_dot, feature_names, threshold):
    lib   = ps.PolynomialLibrary(degree=POLY_DEGREE, include_bias=True)
    opt   = ps.STLSQ(threshold=threshold, alpha=0.05, max_iter=20)
    model = ps.SINDy(feature_library=lib, optimizer=opt)
    model.fit(X, t=1.0, x_dot=X_dot, feature_names=feature_names)
    return model


def score_model(model, X, y_true):
    y_pred   = model.predict(X)[:, TARGET_IDX]
    ss_res   = np.sum((y_true - y_pred) ** 2)
    ss_tot   = np.sum((y_true - y_true.mean()) ** 2)
    r2       = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    coef     = model.coefficients()[TARGET_IDX]
    n_nz     = int(np.count_nonzero(coef))
    return {
        "r2":        round(float(r2), 4),
        "n_nonzero": n_nz,
        "n_total":   coef.size,
        "sparsity":  round(1 - n_nz / coef.size, 4),
        "y_pred":    y_pred,
    }


def grid_search(X_tr, Xd_tr, X_val, y_val):
    """Return (chosen_threshold, sweep_df)."""
    rows = []
    for thr in THRESHOLD_VALUES:
        try:
            m    = build_and_fit(X_tr, Xd_tr, FEATURE_COLS, thr)
            s_tr = score_model(m, X_tr,  X_tr[:, TARGET_IDX])   # train target
            s_v  = score_model(m, X_val, y_val)
            rows.append({"threshold": thr,
                         "r2_train":  s_tr["r2"],
                         "r2_val":    s_v["r2"],
                         "n_nonzero": s_tr["n_nonzero"],
                         "sparsity":  s_tr["sparsity"]})
        except Exception:
            pass
    sweep = pd.DataFrame(rows).dropna()
    if not GRID_SEARCH or sweep.empty:
        return BEST_THRESHOLD, sweep
    best_v  = sweep["r2_val"].max()
    floor   = best_v - R2_TOLERANCE
    cands   = sweep[sweep["r2_val"] >= floor]
    opt_row = cands.loc[cands["n_nonzero"].idxmin()]
    return float(opt_row["threshold"]), sweep


def fit_plot(model, y_true, y_pred, title, path,
             v_splits=None, score=None):
    """Two-panel actual vs predicted + residuals."""
    t = np.arange(len(y_true))
    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                              gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(title, fontsize=9, y=1.01)
    axes[0].plot(t, y_true, label="Actual dI/dt (per 100k)", lw=1.4,
                 color="#1565c0")
    axes[0].plot(t, y_pred, label="SINDy predicted", lw=1.2,
                 color="#e53935", linestyle="--", alpha=0.85)
    axes[0].axhline(0, color="#aaa", lw=0.7, ls=":")
    if v_splits:
        colours = ["#ff9800", "#4caf50"]
        labels  = ["train|val", "val|test"]
        for idx, (x, c, lb) in enumerate(zip(v_splits, colours, labels)):
            axes[0].axvline(x, color=c, lw=1.5, ls="--", label=lb)
    axes[0].set_ylabel("dI/dt (per 100k / week)")
    axes[0].set_xlabel("Time step (weeks)")
    axes[0].legend(fontsize=8, ncol=3)

    res = y_true - y_pred
    if v_splits:
        n0 = v_splits[0]
        n1 = v_splits[1] - v_splits[0]
        axes[1].bar(t[:n0],          res[:n0],       color="#1565c0",
                    alpha=0.4, width=1, label="train")
        axes[1].bar(t[n0:n0+n1],     res[n0:n0+n1],  color="#ff9800",
                    alpha=0.6, width=1, label="val")
        axes[1].bar(t[n0+n1:],       res[n0+n1:],    color="#4caf50",
                    alpha=0.7, width=1, label="test")
        axes[1].legend(fontsize=8, ncol=3)
    else:
        axes[1].bar(t, res, color="#e53935", alpha=0.5, width=1)
    axes[1].axhline(0, color="#555", lw=0.8)
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Time step (weeks)")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# METHOD 1 — Province-by-province, temporal split, shape-weighted R²
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print("METHOD 1 — Province-by-province  (temporal split, weighted R²)")
print(f"{'═'*60}")

m1_rows = []

for rank, prov in enumerate(all_provinces, 1):
    sub = (df[df["province"] == prov]
             .sort_values(["year", "week"])
             .reset_index(drop=True))

    train = sub[sub["year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    val   = sub[sub["year"].isin(VAL_YEARS)].reset_index(drop=True)
    test  = sub[sub["year"].isin(TEST_YEARS)].reset_index(drop=True)

    if len(train) < 20 or len(val) < 4 or len(test) < 4:
        print(f"  [{rank:02d}] {prov} — skipped (insufficient data)")
        continue

    X_tr, Xd_tr, y_tr   = make_Xy(train)
    X_val, _,    y_val   = make_Xy(val)
    X_te,  _,    y_te    = make_Xy(test)

    # shape_score: peak incidence / mean incidence — sharpness of epidemic peaks
    incidence_series = sub["incidence"].values
    shape_score = (incidence_series.max() /
                   (incidence_series.mean() + 1e-9))

    # Grid search on VAL
    thr, sweep = grid_search(X_tr, Xd_tr, X_val, y_val)

    try:
        model   = build_and_fit(X_tr, Xd_tr, FEATURE_COLS, thr)
        s_tr    = score_model(model, X_tr,  y_tr)
        s_val   = score_model(model, X_val, y_val)
        s_te    = score_model(model, X_te,  y_te)
        eq      = model.equations(precision=4)[TARGET_IDX]

        print(f"\n  [{rank:02d}] {prov}")
        print(f"       threshold={thr:.4f}  terms={s_tr['n_nonzero']}  "
              f"shape={shape_score:.2f}")
        print(f"       R² train={s_tr['r2']}  val={s_val['r2']}  "
              f"test={s_te['r2']}")
        print(f"       dI/dt = {eq}")

        # Plot: full timeline (train+val+test concatenated)
        y_all   = np.concatenate([y_tr, y_val, y_te])
        yp_all  = np.concatenate([s_tr["y_pred"],
                                   s_val["y_pred"],
                                   s_te["y_pred"]])
        title = (f"M1 — {prov}  [rank {rank}]  threshold={thr}\n"
                 f"R² train={s_tr['r2']}  val={s_val['r2']}  "
                 f"test={s_te['r2']} ★  terms={s_tr['n_nonzero']}")
        safe = prov.replace("/", "_").replace(" ", "_")
        fit_plot(model, y_all, yp_all, title,
                 os.path.join(M1_DIR, f"{safe}_fit.png"),
                 v_splits=[len(y_tr), len(y_tr) + len(y_val)])

        m1_rows.append({
            "rank":         rank,
            "province":     prov,
            "total_cases":  int(province_totals[prov]),
            "shape_score":  round(shape_score, 4),
            "threshold":    thr,
            "n_nonzero":    s_tr["n_nonzero"],
            "sparsity":     s_tr["sparsity"],
            "r2_train":     s_tr["r2"],
            "r2_val":       s_val["r2"],
            "r2_test":      s_te["r2"],
            "equation":     eq,
        })

    except Exception as e:
        print(f"  [{rank:02d}] {prov} — FAILED: {e}")
        m1_rows.append({
            "rank": rank, "province": prov,
            "total_cases": int(province_totals[prov]),
            "shape_score": round(shape_score, 4),
            "threshold": thr, "n_nonzero": np.nan,
            "sparsity": np.nan, "r2_train": np.nan,
            "r2_val": np.nan, "r2_test": np.nan,
            "equation": f"FAILED: {e}",
        })

m1_df = pd.DataFrame(m1_rows)
m1_df.to_csv(os.path.join(OUT_DIR, "m1_summary.csv"), index=False)

# ── Weighted R² (shape_score weights) ────────────────────────────────────────
valid = m1_df.dropna(subset=["r2_test", "shape_score"])
if len(valid) > 0:
    w        = valid["shape_score"].values
    r2_test  = valid["r2_test"].values
    w_r2     = np.sum(w * r2_test) / np.sum(w)
    plain_r2 = r2_test.mean()

    print(f"\n{'─'*60}")
    print(f"METHOD 1 SUMMARY  ({len(valid)} provinces)")
    print(f"  Plain mean test R²          : {plain_r2:.4f}")
    print(f"  Shape-weighted mean test R² : {w_r2:.4f}")
    print(f"\n  Province breakdown (sorted by shape_score ↓):")
    print(f"  {'Province':<32} {'Shape':>6} {'R²-test':>8} {'Terms':>6}")
    print(f"  {'─'*32} {'─'*6} {'─'*8} {'─'*6}")
    for _, row in valid.sort_values("shape_score", ascending=False).iterrows():
        print(f"  {row['province']:<32} {row['shape_score']:>6.2f} "
              f"{row['r2_test']:>8.4f} {int(row['n_nonzero']):>6}")

# ── Threshold sweep plot (top province) ──────────────────────────────────────
top_prov = all_provinces[0]
sub0 = df[df["province"] == top_prov].sort_values(["year","week"]).reset_index(drop=True)
tr0  = sub0[sub0["year"].isin(TRAIN_YEARS)].reset_index(drop=True)
v0   = sub0[sub0["year"].isin(VAL_YEARS)].reset_index(drop=True)
X_tr0, Xd_tr0, y_tr0 = make_Xy(tr0)
X_v0,  _,      y_v0  = make_Xy(v0)
_, sw0 = grid_search(X_tr0, Xd_tr0, X_v0, y_v0)
if not sw0.empty:
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(sw0["threshold"], sw0["r2_train"], "o-",  ms=3,
             color="#1565c0", label="R² train")
    ax1.plot(sw0["threshold"], sw0["r2_val"],   "^--", ms=3,
             color="#388e3c", label="R² val")
    ax2.plot(sw0["threshold"], sw0["n_nonzero"],"s--", ms=3,
             color="#e53935", label="# terms")
    ax1.set_xlabel("STLSQ threshold")
    ax1.set_ylabel("R²",             color="#1565c0")
    ax2.set_ylabel("Non-zero terms", color="#e53935")
    ax1.set_title(f"M1 threshold sweep — {top_prov}")
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m1_threshold_sweep.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

print(f"\n  M1 results → {OUT_DIR}/m1_summary.csv  +  {M1_DIR}/")


# ════════════════════════════════════════════════════════════════════════════
# METHOD 2 — All 32 provinces pooled, temporal split
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print("METHOD 2 — All provinces pooled  (temporal split)")
print(f"{'═'*60}")

train2 = df[df["year"].isin(TRAIN_YEARS)].reset_index(drop=True)
val2   = df[df["year"].isin(VAL_YEARS)].reset_index(drop=True)
test2  = df[df["year"].isin(TEST_YEARS)].reset_index(drop=True)

print(f"  Pooled sizes — train:{len(train2):,}  val:{len(val2):,}  "
      f"test:{len(test2):,} rows")

X2_tr, Xd2_tr, y2_tr   = make_Xy(train2)
X2_val, _,     y2_val   = make_Xy(val2)
X2_te,  _,     y2_te    = make_Xy(test2)

print(f"\n  Running grid search …")
thr2, sweep2 = grid_search(X2_tr, Xd2_tr, X2_val, y2_val)

if not sweep2.empty:
    best_v2 = sweep2["r2_val"].max()
    print(f"  Best val R²={best_v2:.4f}  →  chosen threshold={thr2:.4f}")

    # Print compact sweep
    print(f"\n  {'Threshold':>10}  {'R²-train':>9}  {'R²-val':>8}  "
          f"{'Terms':>6}  {'Sparsity':>9}")
    print(f"  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*6}  {'─'*9}")
    for _, r in sweep2.iterrows():
        mark = " ◀" if abs(r["threshold"] - thr2) < 1e-6 else ""
        print(f"  {r['threshold']:>10.3f}  {r['r2_train']:>9.4f}  "
              f"{r['r2_val']:>8.4f}  {int(r['n_nonzero']):>6}  "
              f"{r['sparsity']:>9.4f}{mark}")

# Fit final model
m2      = build_and_fit(X2_tr, Xd2_tr, FEATURE_COLS, thr2)
s2_tr   = score_model(m2, X2_tr,  y2_tr)
s2_val  = score_model(m2, X2_val, y2_val)
s2_te   = score_model(m2, X2_te,  y2_te)
eq2     = m2.equations(precision=4)[TARGET_IDX]

print(f"\n  Discovered equation:")
print(f"  dI/dt (per 100k) = {eq2}")
print(f"\n  R² train (2015–2021, all provinces) : {s2_tr['r2']}")
print(f"  R² val   (2022,      all provinces) : {s2_val['r2']}")
print(f"  R² test  (2023,      all provinces) : {s2_te['r2']}  ★")
print(f"  Terms: {s2_tr['n_nonzero']} / {s2_tr['n_total']}  "
      f"(sparsity={s2_tr['sparsity']:.3f})")

# Physical plausibility
coef2     = m2.coefficients()[TARGET_IDX]
lib_names = m2.get_feature_names()
print("\n  Physical plausibility:")
for name, coef in zip(lib_names, coef2):
    if abs(coef) < 1e-10:
        continue
    if "EIP" in name and "incidence" in name:
        tag = "✓ EIP·I → thermal transmission" if coef > 0 else "⚠ check sign"
        print(f"    {name}: {coef:+.4f}  → {tag}")
    elif name == "incidence_lag1":
        tag = "✓ recovery / decay" if coef < 0 else "⚠ positive AR"
        print(f"    {name}: {coef:+.4f}  → {tag}")
    elif "EIP" in name:
        print(f"    {name}: {coef:+.4f}  → EIP thermal term")
    elif "rainfall" in name.lower():
        print(f"    {name}: {coef:+.4f}  → rainfall driver")
    elif "humidity" in name.lower():
        print(f"    {name}: {coef:+.4f}  → humidity driver")

# Per-province R² on test set (2023)
print(f"\n  Per-province R² on TEST set (2023):")
print(f"  {'Rank':>4}  {'Province':<32}  {'R²-test':>8}  {'Cases':>10}")
print(f"  {'─'*4}  {'─'*32}  {'─'*8}  {'─'*10}")
m2_prov_rows = []
for rank, prov in enumerate(all_provinces, 1):
    sub_te = test2[test2["province"] == prov].reset_index(drop=True)
    if len(sub_te) == 0:
        continue
    Xp, _, yp = make_Xy(sub_te)
    sp = score_model(m2, Xp, yp)
    tc = int(province_totals[prov])
    print(f"  {rank:>4}  {prov:<32}  {sp['r2']:>8.4f}  {tc:>10,}")
    m2_prov_rows.append({"rank": rank, "province": prov,
                          "total_cases": tc, "r2_test": sp["r2"]})

# Method 2 per-province fit plots (top 5 only to keep output manageable)
print(f"\n  Generating test-year fit plots for top 5 provinces …")
for prov in all_provinces[:5]:
    sub_all = df[df["province"] == prov].sort_values(["year","week"]).reset_index(drop=True)
    X_all, _, y_all_full = make_Xy(sub_all)
    yp_all = m2.predict(X_all)[:, TARGET_IDX]
    n_tr   = len(sub_all[sub_all["year"].isin(TRAIN_YEARS)])
    n_val  = len(sub_all[sub_all["year"].isin(VAL_YEARS)])
    safe   = prov.replace("/", "_").replace(" ", "_")
    sub_te2 = sub_all[sub_all["year"].isin(TEST_YEARS)].reset_index(drop=True)
    Xp2, _, yp2_true = make_Xy(sub_te2)
    sp2  = score_model(m2, Xp2, yp2_true)
    title = (f"M2 (pooled) — {prov}\n"
             f"threshold={thr2}  R²-test(2023)={sp2['r2']}  "
             f"terms={s2_tr['n_nonzero']}\n"
             f"(single universal equation for all 32 provinces)")
    fit_plot(m2, y_all_full, yp_all, title,
             os.path.join(OUT_DIR, f"m2_{safe}_fit.png"),
             v_splits=[n_tr, n_tr + n_val])

# Save Method 2 summary
m2_summary = {
    "equation":          eq2,
    "threshold":         thr2,
    "n_nonzero":         s2_tr["n_nonzero"],
    "n_total":           s2_tr["n_total"],
    "sparsity":          s2_tr["sparsity"],
    "r2_train_pooled":   s2_tr["r2"],
    "r2_val_pooled":     s2_val["r2"],
    "r2_test_pooled":    s2_te["r2"],
}
pd.DataFrame([m2_summary]).to_csv(
    os.path.join(OUT_DIR, "m2_summary.csv"), index=False)
pd.DataFrame(m2_prov_rows).to_csv(
    os.path.join(OUT_DIR, "m2_per_province_r2.csv"), index=False)

# Threshold sweep plot for Method 2
if not sweep2.empty:
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(sweep2["threshold"], sweep2["r2_train"], "o-",  ms=3,
             color="#1565c0", label="R² train")
    ax1.plot(sweep2["threshold"], sweep2["r2_val"],   "^--", ms=3,
             color="#388e3c", label="R² val")
    ax2.plot(sweep2["threshold"], sweep2["n_nonzero"],"s--", ms=3,
             color="#e53935", label="# terms")
    ax1.axvline(thr2, color="#ff9800", lw=1.5, ls=":",
                label=f"chosen={thr2:.3f}")
    ax1.set_xlabel("STLSQ threshold")
    ax1.set_ylabel("R²",             color="#1565c0")
    ax2.set_ylabel("Non-zero terms", color="#e53935")
    ax1.set_title("M2 threshold sweep — all 32 provinces pooled")
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "m2_threshold_sweep.png"),
                dpi=120, bbox_inches="tight")
    plt.close()

print(f"\n  M2 results → {OUT_DIR}/m2_summary.csv")
print(f"             → {OUT_DIR}/m2_per_province_r2.csv")

# ── Final banner ──────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print("ALL DONE")
print(f"{'═'*60}")
if len(valid) > 0:
    print(f"  M1 shape-weighted R² (test, 2023) : {w_r2:.4f}")
    print(f"  M1 plain mean R²    (test, 2023)  : {plain_r2:.4f}")
print(f"  M2 pooled R²        (test, 2023)  : {s2_te['r2']:.4f}")
print(f"\n  Outputs → {OUT_DIR}/")
