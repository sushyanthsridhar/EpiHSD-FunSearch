"""
followup_analysis.py
────────────────────────────────────────────────────────────────────────────
Post-hoc validation suite for the FunSearch-SINDy two-island result.

Implements six follow-up analyses in priority order:

  1. Data quality filter
     Flag provinces failing zero_pct >= 0.40 OR total_cases < 1000 (train).

  2. Permutation test on island assignments  [HIGHEST VALUE]
     Randomly partition 32 provinces into 13/19 splits 1000 times.
     For each split refit both equation structures and score on test.
     Compare true split score against null distribution.

  3. Cross-equation fitting
     Refit Island A equation structure on every Island B province (and
     vice versa).  If many provinces do equally well under the other
     island's equation, the split is loose.

  4. Spectral decomposition by frequency band
     Break the spectral score into annual, biannual, and high-frequency
     bands.  Tests whether Island A wins on outbreak takeoffs while
     Island B wins on seasonal cycles.

  5. Temperature-stratified equation comparison
     Bin provinces into four temperature bands and compare EIP vs AR
     equation test R2 within each band.

  6. EIP-to-incidence Pearson correlation diagnostic
     Per province, compute r(EIP, incidence) over training years.
     If Island A provinces show higher correlation, the split has
     biological content.  If not, it is likely a structural artifact.

Run:
    cd funsearch/analysis/
    python followup_analysis.py

Outputs: funsearch/results/followup/
    data_quality_filter.csv
    permutation_test_results.csv
    permutation_test_plot.png
    cross_equation_fitting.csv
    cross_equation_plot.png
    spectral_decomposition.csv
    spectral_decomposition_plot.png
    temperature_stratified.csv
    temperature_stratified_plot.png
    eip_incidence_correlation.csv
    eip_incidence_correlation_plot.png
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE          = os.path.dirname(os.path.abspath(__file__))
FUNSEARCH_DIR = os.path.dirname(HERE)
sys.path.insert(0, FUNSEARCH_DIR)

from config import (
    SEIR_CSV, M1_SUMMARY, RESULTS_DIR,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    TARGET_COL, ALL_PROVINCES,
)
from equation import Equation, _term_values
from evaluator import score_province, spectral_score

FOLLOWUP_DIR = os.path.join(RESULTS_DIR, "followup")
os.makedirs(FOLLOWUP_DIR, exist_ok=True)

BREAKDOWN_CSV = os.path.join(RESULTS_DIR, "province_breakdown.csv")
BEST_EQ_CSV   = os.path.join(RESULTS_DIR, "best_equations.csv")

COL_A = "#1565C0"
COL_B = "#2E7D32"

# ── Island definitions (must match config.py _COASTAL) ───────────────────────

ISLAND_A_KEYS = {
    "32 Santo Domingo", "01 Distrito Nacional", "13 La Vega",
    "04 Barahona", "06 Duarte", "24 Sánchez Ramírez",
    "11 La Altagracia", "22 San Juan", "28 Monseñor Nouel",
    "31 San José de Ocoa", "30 Hato Mayor", "20 Samaná", "03 Baoruco",
}
ISLAND_B_KEYS = set(ALL_PROVINCES) - ISLAND_A_KEYS

N_A = len(ISLAND_A_KEYS)   # 13
N_B = len(ISLAND_B_KEYS)   # 19

# ── Province metadata (lat, lon, elev, coast_km, mean_temp, urban, burden) ───

METADATA = {
    "32 Santo Domingo"       : (18.485, -69.931,  30,   2,  27.5, 1.0, 2),
    "01 Distrito Nacional"   : (18.479, -69.890,  14,   3,  28.0, 1.0, 2),
    "13 La Vega"             : (19.221, -70.529, 110,  60,  25.5, 0.5, 2),
    "04 Barahona"            : (18.211, -71.101,  50,   5,  27.8, 0.5, 1),
    "06 Duarte"              : (19.199, -70.033,  80,  55,  26.0, 0.5, 2),
    "24 Sánchez Ramírez"     : (19.052, -70.153, 130,  65,  25.5, 0.0, 1),
    "11 La Altagracia"       : (18.616, -68.712,  20,   8,  28.5, 0.5, 1),
    "22 San Juan"            : (18.806, -71.228, 400,  80,  24.0, 0.5, 1),
    "28 Monseñor Nouel"      : (18.922, -70.415, 200,  70,  25.0, 0.5, 1),
    "31 San José de Ocoa"    : (18.544, -70.503, 550,  50,  23.0, 0.0, 0),
    "30 Hato Mayor"          : (18.764, -69.255, 140,  35,  26.5, 0.0, 1),
    "20 Samaná"              : (19.206, -69.336,  40,   5,  27.0, 0.0, 1),
    "03 Baoruco"             : (18.486, -71.418, 180,  20,  28.8, 0.0, 0),
    "25 Santiago"            : (19.451, -70.697, 170,  90,  26.5, 1.0, 2),
    "21 San Cristóbal"       : (18.418, -70.106,  80,  10,  27.0, 0.5, 1),
    "18 Puerto Plata"        : (19.795, -70.685,  60,   5,  26.0, 0.5, 1),
    "15 Monte Cristi"        : (19.866, -71.648,  20,   5,  28.0, 0.0, 0),
    "29 Monte Plata"         : (18.806, -69.784, 160,  45,  26.0, 0.0, 1),
    "16 Pedernales"          : (17.929, -71.444, 100,  15,  27.5, 0.0, 0),
    "07 Elías Piña"          : (18.875, -71.706, 600, 110,  22.0, 0.0, 0),
    "02 Azua"                : (18.452, -70.735, 120,  25,  28.0, 0.5, 1),
    "05 Dajabón"             : (19.549, -71.707,  80,  30,  26.5, 0.0, 0),
    "08 El Seibo"            : (18.765, -69.038, 200,  30,  26.0, 0.0, 1),
    "09 Espaillat"           : (19.623, -70.275, 250,  40,  24.5, 0.0, 1),
    "10 Independencia"       : (18.407, -71.845, 700,  30,  22.5, 0.0, 0),
    "12 La Romana"           : (18.427, -68.972,  15,   5,  28.0, 0.5, 1),
    "14 Mª Trinidad Sánchez" : (19.374, -69.853,  30,   5,  27.0, 0.0, 1),
    "17 Peravia"             : (18.280, -70.336, 100,  10,  27.5, 0.5, 1),
    "19 Hermanas Mirabal"    : (19.375, -70.307, 200,  50,  25.0, 0.0, 1),
    "23 San Pedro de Macorís": (18.451, -69.301,  15,   3,  27.5, 0.5, 1),
    "26 Santiago Rodríguez"  : (19.481, -71.336, 300,  80,  24.5, 0.0, 0),
    "27 Valverde"            : (19.583, -71.072, 100,  60,  26.0, 0.5, 1),
}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_all():
    print("Loading seir_features.csv ...")
    feat_df = (pd.read_csv(SEIR_CSV)
               .sort_values(["province", "year", "week"])
               .reset_index(drop=True))
    print(f"  {len(feat_df):,} rows, {feat_df['province'].nunique()} provinces")

    print("Loading province_breakdown.csv ...")
    breakdown = pd.read_csv(BREAKDOWN_CSV)
    breakdown["province"] = breakdown["province"].str.strip()

    print("Loading best_equations.csv ...")
    best_eq = pd.read_csv(BEST_EQ_CSV)

    print("Loading m1_summary.csv ...")
    m1 = pd.read_csv(M1_SUMMARY)
    m1["province"] = m1["province"].str.strip()

    return feat_df, breakdown, best_eq, m1


def build_province_dict(feat_df):
    """Return dict: province_name -> full DataFrame (all years)."""
    pdict = {}
    for prov in feat_df["province"].unique():
        pdict[prov] = feat_df[feat_df["province"] == prov].copy()
    return pdict


def parse_best_equations(best_eq_df):
    """
    Parse Island A and B evolved equations from best_equations.csv.
    Returns (eq_a, eq_b) as Equation objects.
    """
    eqs = {}
    for _, row in best_eq_df.iterrows():
        island = row["island"]
        eq_str = str(row["equation"]).strip()
        # Strip the "dI/dt = " prefix produced by Equation.__repr__
        if eq_str.startswith("dI/dt ="):
            eq_str = eq_str[len("dI/dt ="):].strip()
        eqs[island] = Equation.parse(eq_str)
    eq_a = eqs.get("A_coastal") or eqs.get("A") or list(eqs.values())[0]
    eq_b = eqs.get("B_inland")  or eqs.get("B") or list(eqs.values())[1]
    return eq_a, eq_b


# ══════════════════════════════════════════════════════════════════════════════
# Shared utility: OLS coefficient refit on a list of province DataFrames
# ══════════════════════════════════════════════════════════════════════════════

def refit_equation(term_keys, province_dfs_list, train_years):
    """
    Given a set of term names, refit OLS coefficients pooled across
    the supplied province DataFrames on the training years.

    Parameters
    ----------
    term_keys        : iterable of str (library term names)
    province_dfs_list: list of pd.DataFrame
    train_years      : list of int

    Returns
    -------
    Equation with fitted coefficients.
    Returns a zero-coefficient equation on failure.
    """
    terms = list(term_keys)
    if not terms:
        return Equation({})

    X_blocks, y_blocks = [], []
    for df in province_dfs_list:
        subset = df[df["year"].isin(train_years)].copy()
        if len(subset) < max(4, len(terms)):
            continue
        y = subset[TARGET_COL].values.astype(float)
        if not np.all(np.isfinite(y)):
            continue
        try:
            cols = []
            for t in terms:
                v = _term_values(t, subset).astype(float)
                if not np.all(np.isfinite(v)):
                    raise ValueError(f"Non-finite in term {t}")
                cols.append(v)
            X_blocks.append(np.column_stack(cols))
            y_blocks.append(y)
        except Exception:
            continue

    if not X_blocks:
        return Equation({t: 0.0 for t in terms})

    X = np.vstack(X_blocks)
    y = np.clip(np.concatenate(y_blocks), -100.0, 100.0)

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return Equation({t: 0.0 for t in terms})

    if not np.all(np.isfinite(coeffs)):
        return Equation({t: 0.0 for t in terms})

    new_terms = {t: float(c) for t, c in zip(terms, coeffs) if abs(c) > 1e-6}
    return Equation(new_terms) if new_terms else Equation({t: 0.0 for t in terms})


def mean_r2(eq, province_dfs_list, years):
    """Mean R2 across a list of province DataFrames."""
    scores = [score_province(eq, df, years=years) for df in province_dfs_list]
    return float(np.mean(scores)) if scores else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 1: Data quality filter
# ══════════════════════════════════════════════════════════════════════════════

def analysis_data_quality(feat_df):
    """
    Flag provinces failing:
      zero_pct >= 0.40   (40 percent or more of training weeks have 0 cases)
      OR total_cases_train < 1000

    Returns a DataFrame with one row per province and flag columns.
    """
    print("\n" + "="*68)
    print("  ANALYSIS 1: Data quality filter")
    print("="*68)

    train_df = feat_df[feat_df["year"].isin(TRAIN_YEARS)]
    rows = []
    for prov in sorted(feat_df["province"].unique()):
        sub = train_df[train_df["province"] == prov]
        total_cases = sub["total_cases"].sum()
        n_weeks     = len(sub)
        zero_weeks  = (sub["total_cases"] == 0).sum()
        zero_pct    = zero_weeks / n_weeks if n_weeks > 0 else 1.0
        island      = "A_EIP" if prov in ISLAND_A_KEYS else "B_AR"
        fails_zero  = bool(zero_pct >= 0.40)
        fails_count = bool(total_cases < 1000)
        flagged     = fails_zero or fails_count
        rows.append({
            "province"           : prov,
            "island"             : island,
            "total_cases_train"  : int(total_cases),
            "n_train_weeks"      : n_weeks,
            "zero_weeks"         : int(zero_weeks),
            "zero_pct"           : round(zero_pct, 3),
            "fails_zero_pct"     : fails_zero,
            "fails_total_cases"  : fails_count,
            "flagged_for_drop"   : flagged,
        })

    df_out = pd.DataFrame(rows)

    flagged_provs = df_out[df_out["flagged_for_drop"]]["province"].tolist()
    print(f"  Provinces flagged for exclusion ({len(flagged_provs)}):")
    for p in flagged_provs:
        row = df_out[df_out["province"] == p].iloc[0]
        print(f"    {p:<30}  zero_pct={row['zero_pct']:.2f}  "
              f"total_cases={row['total_cases_train']:,}  "
              f"island={row['island']}")

    print(f"\n  Provinces passing filter: "
          f"{(~df_out['flagged_for_drop']).sum()} / {len(df_out)}")

    out_path = os.path.join(FOLLOWUP_DIR, "data_quality_filter.csv")
    df_out.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 2: Permutation test on island assignments
# ══════════════════════════════════════════════════════════════════════════════

def analysis_permutation_test(feat_df, eq_a, eq_b, n_perm=1000, seed=42):
    """
    Null hypothesis: the observed 13/19 split is no better than a random
    partition into groups of the same sizes.

    For each permutation:
      1. Randomly assign 13 provinces to group 1, remaining 19 to group 2.
      2. Refit Island A structure on group 1 training data.
      3. Refit Island B structure on group 2 training data.
      4. Compute weighted mean test R2:
             score = (13/32) * mean_r2(group1) + (19/32) * mean_r2(group2)
    Compare true split score against null distribution.

    Returns
    -------
    dict with null_scores, true_score, p_value
    """
    print("\n" + "="*68)
    print("  ANALYSIS 2: Permutation test on island assignments")
    print("="*68)

    rng = np.random.default_rng(seed)

    pdict    = build_province_dict(feat_df)
    all_provs = list(pdict.keys())
    n_total   = len(all_provs)

    terms_a = set(eq_a.terms.keys())
    terms_b = set(eq_b.terms.keys())

    def combined_score(group1_names, group2_names):
        """Refit both structures and return weighted combined test R2."""
        dfs1 = [pdict[p] for p in group1_names if p in pdict]
        dfs2 = [pdict[p] for p in group2_names if p in pdict]

        if not dfs1 or not dfs2:
            return float("nan")

        fitted_a = refit_equation(terms_a, dfs1, TRAIN_YEARS)
        fitted_b = refit_equation(terms_b, dfs2, TRAIN_YEARS)

        r2_1 = mean_r2(fitted_a, dfs1, TEST_YEARS)
        r2_2 = mean_r2(fitted_b, dfs2, TEST_YEARS)

        w1 = len(dfs1) / (len(dfs1) + len(dfs2))
        w2 = len(dfs2) / (len(dfs1) + len(dfs2))
        return w1 * r2_1 + w2 * r2_2

    # True split score
    true_a_names = [p for p in all_provs if p in ISLAND_A_KEYS]
    true_b_names = [p for p in all_provs if p in ISLAND_B_KEYS]
    true_score   = combined_score(true_a_names, true_b_names)
    print(f"  True split combined test R2: {true_score:.4f}")
    print(f"  Running {n_perm} permutations ...")

    null_scores = []
    for i in range(n_perm):
        if (i + 1) % 200 == 0:
            print(f"    Permutation {i+1}/{n_perm} ...")
        perm = rng.permutation(all_provs)
        g1   = list(perm[:N_A])
        g2   = list(perm[N_A:])
        s    = combined_score(g1, g2)
        if not np.isnan(s):
            null_scores.append(s)

    null_scores = np.array(null_scores)
    p_value     = float(np.mean(null_scores >= true_score))
    null_mean   = float(np.mean(null_scores))
    null_std    = float(np.std(null_scores))
    z_score     = (true_score - null_mean) / (null_std + 1e-12)

    print(f"\n  Null distribution: mean={null_mean:.4f}  std={null_std:.4f}")
    print(f"  True score      : {true_score:.4f}")
    print(f"  z-score         : {z_score:.2f}")
    print(f"  p-value         : {p_value:.4f}")
    if p_value < 0.05:
        print("  CONCLUSION: True split is significantly better than random.")
        print("              The island structure has non-random content.")
    else:
        print("  CONCLUSION: True split is NOT significantly better than random.")
        print("              The island structure may be a search artifact.")

    # Save results
    results_df = pd.DataFrame({
        "null_score": null_scores,
    })
    results_df.to_csv(
        os.path.join(FOLLOWUP_DIR, "permutation_null_distribution.csv"),
        index=False,
    )
    summary_df = pd.DataFrame([{
        "true_score" : round(true_score, 4),
        "null_mean"  : round(null_mean,  4),
        "null_std"   : round(null_std,   4),
        "z_score"    : round(z_score,    2),
        "p_value"    : round(p_value,    4),
        "n_perm"     : n_perm,
        "significant": p_value < 0.05,
    }])
    summary_df.to_csv(
        os.path.join(FOLLOWUP_DIR, "permutation_test_results.csv"),
        index=False,
    )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
    ax.hist(null_scores, bins=50, color="#90A4AE", edgecolor="white",
            label=f"Null distribution (n={n_perm})")
    ax.axvline(true_score, color="#C62828", lw=2.5,
               label=f"True split R2={true_score:.3f}  p={p_value:.3f}")
    ax.axvline(null_mean,  color="#37474F", lw=1.5, linestyle="--",
               label=f"Null mean={null_mean:.3f}")
    ax.set_xlabel("Combined weighted test R2")
    ax.set_ylabel("Count")
    ax.set_title("Permutation test: Is the 13/19 island split better than random?",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FOLLOWUP_DIR, "permutation_test_plot.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved -> followup/permutation_test_results.csv")
    print(f"  Saved -> followup/permutation_test_plot.png")

    return {
        "null_scores" : null_scores,
        "true_score"  : true_score,
        "p_value"     : p_value,
        "z_score"     : z_score,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 3: Cross-equation fitting
# ══════════════════════════════════════════════════════════════════════════════

def analysis_cross_equation(feat_df, eq_a, eq_b, breakdown_df):
    """
    For each province, refit BOTH equation structures (A and B) using only
    that province's own training data (single-province fit).
    Compare test R2 under native vs foreign equation structure.

    A large gap means the province strongly prefers its native equation.
    A small gap means the province is indifferent and the split is loose.
    """
    print("\n" + "="*68)
    print("  ANALYSIS 3: Cross-equation fitting")
    print("="*68)

    pdict   = build_province_dict(feat_df)
    terms_a = set(eq_a.terms.keys())
    terms_b = set(eq_b.terms.keys())

    rows = []
    for prov, df in pdict.items():
        island   = "A_EIP" if prov in ISLAND_A_KEYS else "B_AR"
        df_list  = [df]

        fitted_a = refit_equation(terms_a, df_list, TRAIN_YEARS)
        fitted_b = refit_equation(terms_b, df_list, TRAIN_YEARS)

        r2_a = score_province(fitted_a, df, years=TEST_YEARS)
        r2_b = score_province(fitted_b, df, years=TEST_YEARS)

        native_r2  = r2_a if island == "A_EIP" else r2_b
        foreign_r2 = r2_b if island == "A_EIP" else r2_a
        preference = native_r2 - foreign_r2

        rows.append({
            "province"       : prov,
            "island"         : island,
            "r2_eq_A"        : round(r2_a,       4),
            "r2_eq_B"        : round(r2_b,       4),
            "native_r2"      : round(native_r2,  4),
            "foreign_r2"     : round(foreign_r2, 4),
            "native_minus_foreign": round(preference, 4),
            "prefers_native" : preference > 0,
        })

    df_out = pd.DataFrame(rows)

    # Summary statistics
    a_prov = df_out[df_out["island"] == "A_EIP"]
    b_prov = df_out[df_out["island"] == "B_AR"]

    print(f"\n  Island A provinces ({len(a_prov)}):")
    print(f"    Mean R2 under A equation (native) : "
          f"{a_prov['native_r2'].mean():.4f}")
    print(f"    Mean R2 under B equation (foreign): "
          f"{a_prov['foreign_r2'].mean():.4f}")
    print(f"    Mean preference (native - foreign): "
          f"{a_prov['native_minus_foreign'].mean():.4f}")
    print(f"    Provinces preferring native       : "
          f"{a_prov['prefers_native'].sum()} / {len(a_prov)}")

    print(f"\n  Island B provinces ({len(b_prov)}):")
    print(f"    Mean R2 under B equation (native) : "
          f"{b_prov['native_r2'].mean():.4f}")
    print(f"    Mean R2 under A equation (foreign): "
          f"{b_prov['foreign_r2'].mean():.4f}")
    print(f"    Mean preference (native - foreign): "
          f"{b_prov['native_minus_foreign'].mean():.4f}")
    print(f"    Provinces preferring native       : "
          f"{b_prov['prefers_native'].sum()} / {len(b_prov)}")

    # Welch t-test: do provinces significantly prefer their native equation?
    t_a, p_a = stats.ttest_1samp(
        a_prov["native_minus_foreign"].dropna(), 0)
    t_b, p_b = stats.ttest_1samp(
        b_prov["native_minus_foreign"].dropna(), 0)
    print(f"\n  One-sample t-test (preference > 0):")
    print(f"    Island A: t={t_a:.2f}  p={p_a:.4f}")
    print(f"    Island B: t={t_b:.2f}  p={p_b:.4f}")

    out_path = os.path.join(FOLLOWUP_DIR, "cross_equation_fitting.csv")
    df_out.to_csv(out_path, index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

    for ax, grp, title, col in zip(
            axes,
            [a_prov, b_prov],
            ["Island A provinces under A vs B equation",
             "Island B provinces under B vs A equation"],
            [COL_A, COL_B]):

        native_col  = "r2_eq_A" if col == COL_A else "r2_eq_B"
        foreign_col = "r2_eq_B" if col == COL_A else "r2_eq_A"
        names = [p.split(None, 1)[1] if " " in p else p
                 for p in grp["province"]]

        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w/2, grp[native_col],  width=w,
               color=col,     alpha=0.85, label="Native equation")
        ax.bar(x + w/2, grp[foreign_col], width=w,
               color="#78909C", alpha=0.75, label="Foreign equation")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Test R2")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Cross-equation fitting: native vs foreign equation structure",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FOLLOWUP_DIR, "cross_equation_plot.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved -> followup/cross_equation_fitting.csv")
    print(f"  Saved -> followup/cross_equation_plot.png")
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 4: Spectral decomposition by frequency band
# ══════════════════════════════════════════════════════════════════════════════

def spectral_by_band(eq, df, years, n_bands=5):
    """
    Compute per-frequency-bin spectral score components.
    Returns array of length n_bands: score at each harmonic of the series.

    For a 260-week training period (5 years):
      bin 1 -> period = 260 weeks (5-year trend)
      bin 2 -> period = 130 weeks (2.5-year)
      bin 4 -> period =  65 weeks (slightly > annual)
      bin 5 -> period =  52 weeks (annual — key dengue signal)
      bin 10 -> period = 26 weeks (biannual)

    For a 104-week val/test period (2 years):
      bin 2 -> period = 52 weeks (annual)
      bin 4 -> period = 26 weeks (biannual)
    """
    subset = df[df["year"].isin(years)].copy()
    if len(subset) < 2 * n_bands + 2:
        return np.full(n_bands, np.nan)

    y_true = subset[TARGET_COL].values.astype(float)
    try:
        y_pred = eq.evaluate(subset)
    except Exception:
        return np.full(n_bands, np.nan)

    if not np.all(np.isfinite(y_pred)):
        return np.full(n_bands, np.nan)

    y_pred = np.clip(y_pred, -1000.0, 1000.0)

    psd_true = np.abs(np.fft.rfft(y_true - np.mean(y_true))) ** 2
    psd_pred = np.abs(np.fft.rfft(y_pred - np.mean(y_pred))) ** 2

    bands = []
    for k in range(1, n_bands + 1):
        if k >= len(psd_true) or k >= len(psd_pred):
            bands.append(np.nan)
            continue
        true_sum = psd_true[k] / (psd_true[1:].sum() + 1e-12)
        pred_sum = psd_pred[k] / (psd_pred[1:].sum() + 1e-12)
        score = float(np.clip(1.0 - abs(true_sum - pred_sum), 0.0, 1.0))
        bands.append(score)
    return np.array(bands)


def analysis_spectral_decomposition(feat_df, eq_a, eq_b):
    """
    For every province, compute the per-band spectral score under both
    Island A and Island B equations on the test period.

    Key bands for a 104-week test series:
      band 2 -> annual cycle (period = 52 weeks)
      band 4 -> biannual (period = 26 weeks)
      band 1 -> interannual trend
      bands 3,5 -> intermediate harmonics
    """
    print("\n" + "="*68)
    print("  ANALYSIS 4: Spectral decomposition by frequency band")
    print("="*68)

    pdict = build_province_dict(feat_df)

    n_bands   = 5
    band_periods_test = [f"period_{int(104/(k))}wk" for k in range(1, n_bands+1)]

    rows = []
    for prov, df in pdict.items():
        island = "A_EIP" if prov in ISLAND_A_KEYS else "B_AR"

        bands_a = spectral_by_band(eq_a, df, TEST_YEARS, n_bands=n_bands)
        bands_b = spectral_by_band(eq_b, df, TEST_YEARS, n_bands=n_bands)

        row = {"province": prov, "island": island}
        for k, (ba, bb) in enumerate(zip(bands_a, bands_b)):
            label = band_periods_test[k]
            row[f"eq_A_{label}"] = round(float(ba), 4) if not np.isnan(ba) else np.nan
            row[f"eq_B_{label}"] = round(float(bb), 4) if not np.isnan(bb) else np.nan
        rows.append(row)

    df_out = pd.DataFrame(rows)

    # Summary: Island A vs Island B mean score per band per equation
    print(f"\n  {'Band':<22} {'eq_A mean':>10} {'eq_B mean':>10}  "
          f"{'A-B':>8}  {'note'}")
    print("  " + "-"*65)

    band_labels_display = [
        "interannual (104wk)",
        "annual (52wk)",
        "sub-annual (34wk)",
        "biannual (26wk)",
        "tertiary (20wk)",
    ]
    for k in range(n_bands):
        period_key = band_periods_test[k]
        col_a = f"eq_A_{period_key}"
        col_b = f"eq_B_{period_key}"
        mean_a = df_out[col_a].mean()
        mean_b = df_out[col_b].mean()
        diff   = mean_a - mean_b
        winner = "A wins" if diff > 0.02 else ("B wins" if diff < -0.02 else "tied")
        label  = band_labels_display[k]
        print(f"  {label:<22} {mean_a:>10.4f} {mean_b:>10.4f}  "
              f"{diff:>+8.4f}  {winner}")

    out_path = os.path.join(FOLLOWUP_DIR, "spectral_decomposition.csv")
    df_out.to_csv(out_path, index=False)

    # Plot
    fig, axes = plt.subplots(1, n_bands, figsize=(16, 4), facecolor="white")
    for k, ax in enumerate(axes):
        period_key = band_periods_test[k]
        col_a = f"eq_A_{period_key}"
        col_b = f"eq_B_{period_key}"
        label = band_labels_display[k]

        a_vals = df_out.loc[df_out["island"] == "A_EIP", col_a].dropna()
        b_vals = df_out.loc[df_out["island"] == "B_AR",  col_b].dropna()

        ax.boxplot(
            [a_vals, b_vals],
            labels=["Island A\n(EIP eq)", "Island B\n(AR eq)"],
            patch_artist=True,
            boxprops=dict(facecolor="#E3F2FD"),
            medianprops=dict(color="navy", lw=2),
        )
        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.set_ylabel("Spectral score" if k == 0 else "")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle(
        "Spectral score per frequency band: each island scored under its OWN equation",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(os.path.join(FOLLOWUP_DIR, "spectral_decomposition_plot.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved -> followup/spectral_decomposition.csv")
    print(f"  Saved -> followup/spectral_decomposition_plot.png")
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 5: Temperature-stratified equation comparison
# ══════════════════════════════════════════════════════════════════════════════

def analysis_temperature_stratified(feat_df, eq_a, eq_b):
    """
    Bin provinces into four temperature bands and compare:
      - R2 under Island A equation (EIP-bearing)
      - R2 under Island B equation (autoregressive)

    Each province is scored individually using its own province data
    with the group-level equations (no re-fitting).
    This directly tests whether the EIP equation consistently outperforms
    the AR equation in warmer temperature regimes.
    """
    print("\n" + "="*68)
    print("  ANALYSIS 5: Temperature-stratified equation comparison")
    print("="*68)

    pdict = build_province_dict(feat_df)
    bins  = [0, 24, 26, 28, 100]
    labels_bin = ["<24C", "24-26C", "26-28C", ">28C"]

    rows = []
    for prov, df in pdict.items():
        meta   = METADATA.get(prov)
        if meta is None:
            continue
        _, _, _, _, mean_temp, _, _ = meta
        island = "A_EIP" if prov in ISLAND_A_KEYS else "B_AR"

        r2_a = score_province(eq_a, df, years=TEST_YEARS)
        r2_b = score_province(eq_b, df, years=TEST_YEARS)

        temp_band = pd.cut([mean_temp], bins=bins, labels=labels_bin)[0]

        rows.append({
            "province"  : prov,
            "island"    : island,
            "mean_temp" : mean_temp,
            "temp_band" : str(temp_band),
            "r2_eq_A"   : round(r2_a, 4),
            "r2_eq_B"   : round(r2_b, 4),
            "A_minus_B" : round(r2_a - r2_b, 4),
        })

    df_out = pd.DataFrame(rows)

    print(f"\n  {'Temp band':<10} {'n':>4}  {'mean r2 Eq-A':>13} "
          f"{'mean r2 Eq-B':>13}  {'A-B':>8}  {'EIP wins?'}")
    print("  " + "-"*65)
    for band in labels_bin:
        sub = df_out[df_out["temp_band"] == band]
        if len(sub) == 0:
            continue
        mean_a = sub["r2_eq_A"].mean()
        mean_b = sub["r2_eq_B"].mean()
        diff   = mean_a - mean_b
        winner = "EIP wins" if diff > 0.02 else ("AR wins" if diff < -0.02 else "tied")
        print(f"  {band:<10} {len(sub):>4}  {mean_a:>13.4f} "
              f"{mean_b:>13.4f}  {diff:>+8.4f}  {winner}")

    out_path = os.path.join(FOLLOWUP_DIR, "temperature_stratified.csv")
    df_out.to_csv(out_path, index=False)

    # Plot
    fig, axes = plt.subplots(1, len(labels_bin), figsize=(14, 4),
                             sharey=True, facecolor="white")
    for ax, band in zip(axes, labels_bin):
        sub = df_out[df_out["temp_band"] == band]
        if len(sub) == 0:
            ax.set_title(band, fontsize=9)
            continue
        colors = [COL_A if r >= 0 else COL_B
                  for r in sub["A_minus_B"]]
        ax.bar(range(len(sub)), sub["r2_eq_A"], color=COL_A,
               alpha=0.7, label="EIP eq (A)")
        ax.bar(range(len(sub)), sub["r2_eq_B"], color=COL_B,
               alpha=0.45, label="AR eq (B)")
        ax.axhline(0, color="black", lw=0.7)
        pnames = [p.split(None, 1)[1][:8] if " " in p else p[:8]
                  for p in sub["province"]]
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(pnames, rotation=60, ha="right", fontsize=6)
        ax.set_title(f"{band}\n(n={len(sub)})", fontsize=8, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Test R2")
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Temperature-stratified comparison: EIP equation vs AR equation\n"
        "(both are group-level equations scored per province without re-fitting)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(os.path.join(FOLLOWUP_DIR, "temperature_stratified_plot.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved -> followup/temperature_stratified.csv")
    print(f"  Saved -> followup/temperature_stratified_plot.png")
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 6: EIP-to-incidence Pearson correlation diagnostic
# ══════════════════════════════════════════════════════════════════════════════

def analysis_eip_incidence_correlation(feat_df):
    """
    For each province, compute Pearson r between weekly EIP and weekly
    incidence over the training period.

    If Island A provinces show systematically higher r(EIP, incidence)
    than Island B provinces, there is direct observational evidence that
    the island split reflects biological content.

    If the correlations are indistinguishable, the split is likely a
    structural artifact of the search trajectory.
    """
    print("\n" + "="*68)
    print("  ANALYSIS 6: EIP-to-incidence Pearson correlation")
    print("="*68)

    train_df = feat_df[feat_df["year"].isin(TRAIN_YEARS)]
    rows = []

    for prov in sorted(feat_df["province"].unique()):
        sub    = train_df[train_df["province"] == prov]
        island = "A_EIP" if prov in ISLAND_A_KEYS else "B_AR"
        meta   = METADATA.get(prov)
        temp   = meta[4] if meta else np.nan

        if len(sub) < 10:
            continue

        eip       = sub["EIP"].values.astype(float)
        incidence = sub["incidence"].values.astype(float)

        mask = np.isfinite(eip) & np.isfinite(incidence)
        if mask.sum() < 10:
            continue

        r, p = stats.pearsonr(eip[mask], incidence[mask])

        # Also compute lagged correlations (EIP leads incidence by 2-4 weeks)
        eip_lag2 = sub["EIP"].shift(-2).values.astype(float)
        eip_lag4 = sub["EIP"].shift(-4).values.astype(float)
        mask2    = np.isfinite(eip_lag2) & np.isfinite(incidence)
        mask4    = np.isfinite(eip_lag4) & np.isfinite(incidence)

        r_lag2 = stats.pearsonr(eip_lag2[mask2], incidence[mask2])[0] \
                 if mask2.sum() >= 10 else np.nan
        r_lag4 = stats.pearsonr(eip_lag4[mask4], incidence[mask4])[0] \
                 if mask4.sum() >= 10 else np.nan

        rows.append({
            "province"        : prov,
            "island"          : island,
            "mean_temp"       : temp,
            "r_EIP_incidence" : round(float(r), 4),
            "p_value"         : round(float(p), 4),
            "r_EIP_lag2"      : round(float(r_lag2), 4) if not np.isnan(r_lag2) else np.nan,
            "r_EIP_lag4"      : round(float(r_lag4), 4) if not np.isnan(r_lag4) else np.nan,
            "max_lagged_r"    : round(float(np.nanmax([r, r_lag2, r_lag4])), 4),
        })

    df_out = pd.DataFrame(rows)

    a_vals = df_out.loc[df_out["island"] == "A_EIP", "r_EIP_incidence"].dropna()
    b_vals = df_out.loc[df_out["island"] == "B_AR",  "r_EIP_incidence"].dropna()

    t_stat, p_welch = stats.ttest_ind(a_vals, b_vals, equal_var=False)

    print(f"\n  Island A  mean r(EIP,incidence) = {a_vals.mean():.4f}  "
          f"std={a_vals.std():.4f}  n={len(a_vals)}")
    print(f"  Island B  mean r(EIP,incidence) = {b_vals.mean():.4f}  "
          f"std={b_vals.std():.4f}  n={len(b_vals)}")
    print(f"\n  Welch t-test (A vs B): t={t_stat:.2f}  p={p_welch:.4f}")
    if p_welch < 0.05:
        print("  CONCLUSION: Island A provinces have significantly higher "
              "EIP-incidence correlation.")
        print("              The island split has direct biological support.")
    else:
        print("  CONCLUSION: EIP-incidence correlation does NOT significantly "
              "differ between islands.")
        print("              The split may be a structural artifact.")

    print("\n  Top 10 provinces by r(EIP, incidence):")
    top = df_out.nlargest(10, "r_EIP_incidence")[
        ["province", "island", "mean_temp", "r_EIP_incidence", "p_value"]]
    print(top.to_string(index=False))

    out_path = os.path.join(FOLLOWUP_DIR, "eip_incidence_correlation.csv")
    df_out.to_csv(out_path, index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")

    # Left: strip chart of r by island
    ax = axes[0]
    for island, color, jitter_x in [("A_EIP", COL_A, -0.12), ("B_AR", COL_B, 0.12)]:
        sub   = df_out[df_out["island"] == island]["r_EIP_incidence"].dropna()
        xpos  = 0 if island == "A_EIP" else 1
        xs    = np.full(len(sub), xpos) + np.random.default_rng(0).uniform(
                    -0.08, 0.08, len(sub))
        ax.scatter(xs, sub, color=color, alpha=0.7, s=50, zorder=3)
        ax.plot([xpos - 0.25, xpos + 0.25], [sub.mean(), sub.mean()],
                color=color, lw=3, zorder=4)

    ax.axhline(0, color="#aaa", lw=0.8, linestyle="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Island A\n(EIP equation)", "Island B\n(AR equation)"])
    ax.set_ylabel("Pearson r (EIP vs incidence, training years)")
    ax.set_title("EIP-incidence correlation by island\n"
                 f"(Welch t p={p_welch:.3f})",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    patch_a = mpatches.Patch(color=COL_A, label=f"Island A  mean={a_vals.mean():.3f}")
    patch_b = mpatches.Patch(color=COL_B, label=f"Island B  mean={b_vals.mean():.3f}")
    ax.legend(handles=[patch_a, patch_b], fontsize=8)

    # Right: scatter r vs mean temperature, coloured by island
    ax2 = axes[1]
    for island, color, marker in [("A_EIP", COL_A, "*"), ("B_AR", COL_B, "o")]:
        sub = df_out[df_out["island"] == island]
        ax2.scatter(sub["mean_temp"], sub["r_EIP_incidence"],
                    color=color, marker=marker,
                    s=100 if marker == "*" else 60,
                    alpha=0.8, label=island, zorder=3)
        for _, row in sub.iterrows():
            short = str(row["province"]).split(None, 1)[-1][:8]
            ax2.annotate(short, (row["mean_temp"], row["r_EIP_incidence"]),
                         fontsize=5.5, xytext=(3, 3),
                         textcoords="offset points", color=color)

    ax2.axhline(0, color="#aaa", lw=0.8, linestyle="--")
    ax2.set_xlabel("Province mean temperature (C)")
    ax2.set_ylabel("r(EIP, incidence)")
    ax2.set_title("EIP-incidence r vs temperature\n"
                  "(expected: higher r at warmer temps for EIP provinces)",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FOLLOWUP_DIR, "eip_incidence_correlation_plot.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved -> followup/eip_incidence_correlation.csv")
    print(f"  Saved -> followup/eip_incidence_correlation_plot.png")
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 68)
    print("  FunSearch-SINDy: Follow-up Validation Suite")
    print("  All 6 post-hoc analyses")
    print("=" * 68)

    feat_df, breakdown_df, best_eq_df, m1_df = load_all()
    eq_a, eq_b = parse_best_equations(best_eq_df)

    print(f"\n  Island A equation ({eq_a.n_terms()} terms): {eq_a}")
    print(f"  Island B equation ({eq_b.n_terms()} terms): {eq_b}")

    # ── Run all analyses ──────────────────────────────────────────────────────

    dq_df   = analysis_data_quality(feat_df)

    perm    = analysis_permutation_test(
                  feat_df, eq_a, eq_b, n_perm=1000, seed=42)

    cross   = analysis_cross_equation(feat_df, eq_a, eq_b, breakdown_df)

    spec    = analysis_spectral_decomposition(feat_df, eq_a, eq_b)

    temp    = analysis_temperature_stratified(feat_df, eq_a, eq_b)

    eip_cor = analysis_eip_incidence_correlation(feat_df)

    # ── Print master summary ──────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  MASTER SUMMARY")
    print("=" * 68)

    n_flagged = dq_df["flagged_for_drop"].sum()
    print(f"\n  1. Data quality filter: {n_flagged} provinces flagged "
          f"({n_flagged - dq_df[dq_df['flagged_for_drop'] & (dq_df['island']=='A_EIP')].shape[0]} B, "
          f"{dq_df[dq_df['flagged_for_drop'] & (dq_df['island']=='A_EIP')].shape[0]} A)")

    print(f"\n  2. Permutation test: true score={perm['true_score']:.4f}  "
          f"z={perm['z_score']:.2f}  p={perm['p_value']:.4f}  "
          f"{'SIGNIFICANT' if perm['p_value'] < 0.05 else 'NOT SIGNIFICANT'}")

    a_pref = cross[cross["island"] == "A_EIP"]["prefers_native"].mean()
    b_pref = cross[cross["island"] == "B_AR"]["prefers_native"].mean()
    print(f"\n  3. Cross-equation: Island A prefers native in "
          f"{a_pref*100:.0f}% of provinces, "
          f"Island B in {b_pref*100:.0f}%")

    print(f"\n  4. Spectral decomposition: see "
          f"followup/spectral_decomposition_plot.png")

    hot_sub  = temp[temp["temp_band"] == ">28C"]
    hot_diff = hot_sub["A_minus_B"].mean() if len(hot_sub) > 0 else float("nan")
    print(f"\n  5. Temperature stratification: EIP minus AR in >28C band = "
          f"{hot_diff:+.4f}")

    from scipy import stats as _stats
    a_r = eip_cor[eip_cor["island"] == "A_EIP"]["r_EIP_incidence"].dropna()
    b_r = eip_cor[eip_cor["island"] == "B_AR"]["r_EIP_incidence"].dropna()
    _, p_eip = _stats.ttest_ind(a_r, b_r, equal_var=False)
    print(f"\n  6. EIP-incidence r: Island A mean={a_r.mean():.4f}  "
          f"Island B mean={b_r.mean():.4f}  p={p_eip:.4f}  "
          f"{'SIGNIFICANT' if p_eip < 0.05 else 'NOT SIGNIFICANT'}")

    print(f"\n  All outputs written to: {FOLLOWUP_DIR}")
    print("  Done.")


if __name__ == "__main__":
    main()
