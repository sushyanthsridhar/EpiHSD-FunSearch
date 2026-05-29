"""
run_ga_sindy.py
─────────────────────────────────────────────────────────────────────────────
Main entry point for funsearch-ga-sindy.

Architecture
────────────
  N_ISLANDS parallel evolutionary islands, each searching independently
  over the 66-term equation library.  All islands train on ALL
  quality-passing provinces (no province-to-island assignment).

  A GA coordinator runs above the islands.  Every epoch:
    1. Each island evolves for GENERATIONS_PER_EPOCH generations.
    2. GA selects the two best island champions by tournament.
    3. Crossover produces an offspring.
    4. LLM mutator (algorithmic or Claude API) generates N variants.
    5. Best variant is OLS-fitted and injected into all islands.

  Data quality filter (applied before evolution):
    Drop provinces with zero_pct >= 0.40 OR total_cases_train < 1000.

  Chronological data split (within each surviving province):
    Train : TRAIN_YEARS  (fed province by province in order)
    Val   : VAL_YEARS    (fitness signal throughout evolution)
    Test  : TEST_YEARS   (held out, evaluated only at the end)

Usage
─────
  cd funsearch-ga-sindy/
  python run_ga_sindy.py

Outputs (written to funsearch-ga-sindy/results/)
────────
  best_equation.csv        — final equation, per-province test R²
  epoch_history.csv        — global best fitness per epoch
  province_breakdown.csv   — R² and spectral score on train/val/test
  final_summary.txt        — human-readable summary
  run_ga_sindy.log         — full console log
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
import time
import logging
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import (
    SEIR_CSV, M1_SUMMARY, RESULTS_DIR,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    N_ISLANDS, MAX_ZERO_PCT, MIN_TOTAL_CASES,
    compute_max_islands,
)
from equation import Equation
from evaluator import score_province, spectral_score
from sindy_fitter import refit, data_budget_report
from island import Island
from ga_coordinator import GACoordinator


# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_ga_sindy.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading and quality filter
# ══════════════════════════════════════════════════════════════════════════════

def load_and_filter():
    log.info("=" * 68)
    log.info("  funsearch-ga-sindy")
    log.info("  Multi-island GA-SINDy with LLM structural mutation")
    log.info("=" * 68)

    log.info("\n── Loading seir_features.csv ...")
    feat_df = (pd.read_csv(SEIR_CSV)
               .sort_values(["province", "year", "week"])
               .reset_index(drop=True))
    log.info(f"   {len(feat_df):,} rows  |  "
             f"{feat_df['province'].nunique()} provinces")

    log.info("── Loading m1_summary.csv (SINDy M1 baseline for seeding) ...")
    m1 = pd.read_csv(M1_SUMMARY)
    m1["province"] = m1["province"].str.strip()
    sindy_r2 = dict(zip(m1["province"], m1["r2_test"]))
    eq_lookup = dict(zip(m1["province"], m1["equation"].str.strip()))
    log.info(f"   {len(eq_lookup)} province equations  |  "
             f"SINDy mean test R²={m1['r2_test'].mean():.4f}")

    # ── Apply data quality filter ─────────────────────────────────────────────
    log.info(f"\n── Data quality filter "
             f"(zero_pct < {MAX_ZERO_PCT:.0%}  AND  "
             f"total_cases_train >= {MIN_TOTAL_CASES:,}) ...")

    train_df = feat_df[feat_df["year"].isin(TRAIN_YEARS)]
    passing_provs = []
    dropped_provs = []

    for prov in sorted(feat_df["province"].unique()):
        sub        = train_df[train_df["province"] == prov]
        total_cases = sub["total_cases"].sum()
        n_weeks     = len(sub)
        zero_pct    = (sub["total_cases"] == 0).sum() / n_weeks if n_weeks > 0 else 1.0

        if zero_pct < MAX_ZERO_PCT and total_cases >= MIN_TOTAL_CASES:
            passing_provs.append(prov)
        else:
            dropped_provs.append((prov, zero_pct, total_cases))

    log.info(f"   Passing : {len(passing_provs)} provinces")
    log.info(f"   Dropped : {len(dropped_provs)} provinces")
    for prov, zp, tc in dropped_provs:
        log.info(f"     {prov:<32}  zero_pct={zp:.2f}  cases={tc:,}")

    if len(passing_provs) < 2:
        raise RuntimeError("Fewer than 2 provinces survived the filter. "
                           "Check your data or relax thresholds.")

    # ── Build province DataFrames ─────────────────────────────────────────────
    province_dfs = []
    for prov in passing_provs:
        df = feat_df[feat_df["province"] == prov].copy()
        province_dfs.append(df)

    log.info(f"\n   Provinces used for evolution ({len(province_dfs)}):")
    for p in passing_provs:
        sindy = sindy_r2.get(p, float("nan"))
        log.info(f"     {p}  (SINDy baseline R²={sindy:.3f})")

    # ── Data budget report ────────────────────────────────────────────────────
    budget = data_budget_report(province_dfs, TRAIN_YEARS)
    max_info = compute_max_islands(
        n_provinces=len(province_dfs),
        train_years=TRAIN_YEARS,
    )
    log.info(f"\n── Data budget:")
    log.info(f"   Total training rows  : {budget['total_train_rows']:,}")
    log.info(f"   Mean rows / province : {budget['mean_rows_per_province']:.0f}")
    log.info(f"   Data-theoretic max islands: {max_info['data_theoretical_max']}")
    log.info(f"   Practical recommended max : {max_info['practical_recommended_max']}")
    log.info(f"   Running with N_ISLANDS    : {N_ISLANDS}")

    return province_dfs, passing_provs, eq_lookup, sindy_r2


# ══════════════════════════════════════════════════════════════════════════════
# Seed builder (union of SINDy M1 terms from all passing provinces)
# ══════════════════════════════════════════════════════════════════════════════

def build_seed_terms(passing_provs, eq_lookup):
    """Union of all SINDy M1 term names for the quality-passing provinces."""
    all_terms = set()
    for prov in passing_provs:
        eq_str = eq_lookup.get(prov, "")
        if not eq_str:
            continue
        try:
            prefix = "dI/dt ="
            if eq_str.strip().startswith(prefix):
                eq_str = eq_str.strip()[len(prefix):].strip()
            eq = Equation.parse(eq_str)
            all_terms |= set(eq.terms.keys())
        except Exception:
            pass
    log.info(f"\n── Seed terms (union of SINDy M1, {len(all_terms)} terms):")
    log.info(f"   {sorted(all_terms)}")
    return all_terms


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def province_breakdown(equation, province_dfs, province_names, sindy_r2):
    rows = []
    for prov, df in zip(province_names, province_dfs):
        r2_tr = score_province(equation, df, years=TRAIN_YEARS)
        r2_v  = score_province(equation, df, years=VAL_YEARS)
        r2_te = score_province(equation, df, years=TEST_YEARS)
        sp_tr = spectral_score(equation, df, years=TRAIN_YEARS)
        sp_v  = spectral_score(equation, df, years=VAL_YEARS)
        sp_te = spectral_score(equation, df, years=TEST_YEARS)
        s_te  = sindy_r2.get(prov, float("nan"))
        rows.append({
            "province"       : prov,
            "fs_r2_train"    : round(r2_tr, 4),
            "fs_r2_val"      : round(r2_v,  4),
            "fs_r2_test"     : round(r2_te, 4),
            "fs_spec_train"  : round(sp_tr, 4),
            "fs_spec_val"    : round(sp_v,  4),
            "fs_spec_test"   : round(sp_te, 4),
            "sindy_r2_test"  : round(s_te,  4) if not np.isnan(s_te) else float("nan"),
            "delta_test"     : round(r2_te - s_te, 4) if not np.isnan(s_te) else float("nan"),
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    province_dfs, passing_provs, eq_lookup, sindy_r2 = load_and_filter()
    seed_terms = build_seed_terms(passing_provs, eq_lookup)

    # ── Build islands ─────────────────────────────────────────────────────────
    log.info(f"\n── Initialising {N_ISLANDS} islands ...")
    islands = []
    for i in range(N_ISLANDS):
        island = Island(
            name         = f"Island_{i+1}",
            province_dfs = province_dfs,
            seed_terms   = seed_terms,
        )
        islands.append(island)

    # ── Run GA coordinator ────────────────────────────────────────────────────
    coordinator = GACoordinator(
        islands      = islands,
        province_dfs = province_dfs,
    )

    best_eq = coordinator.run(verbose=True)

    # ── Final evaluation on test set ──────────────────────────────────────────
    log.info(f"\n{'='*68}")
    log.info("  FINAL EVALUATION (test years — held-out)")
    log.info(f"{'='*68}")

    prov_rows = province_breakdown(best_eq, province_dfs, passing_provs, sindy_r2)

    mean_test   = np.nanmean([r["fs_r2_test"]   for r in prov_rows])
    mean_val    = np.nanmean([r["fs_r2_val"]     for r in prov_rows])
    mean_train  = np.nanmean([r["fs_r2_train"]   for r in prov_rows])
    mean_sindy  = np.nanmean([r["sindy_r2_test"] for r in prov_rows
                              if not np.isnan(r["sindy_r2_test"])])
    mean_delta  = mean_test - mean_sindy

    log.info(f"\n  Best equation ({best_eq.n_terms()} terms):")
    log.info(f"  {best_eq}")
    log.info(f"\n  {'Metric':<30} {'Value':>10}")
    log.info(f"  {'─'*42}")
    log.info(f"  {'mean R² train':<30} {mean_train:>10.4f}")
    log.info(f"  {'mean R² val':<30} {mean_val:>10.4f}")
    log.info(f"  {'mean R² test (GA-SINDy)':<30} {mean_test:>10.4f}")
    log.info(f"  {'mean R² test (SINDy M1)':<30} {mean_sindy:>10.4f}")
    log.info(f"  {'Δ (GA-SINDy − SINDy M1)':<30} {mean_delta:>+10.4f}")

    log.info(f"\n  {'Province':<32} {'Tr':>6} {'Val':>6} "
             f"{'Test':>6}  {'SINDy':>6} {'Δ':>7}  {'Spec_te':>8}")
    log.info(f"  {'─'*80}")
    for r in prov_rows:
        name = r["province"].split(None, 1)[1] if " " in r["province"] else r["province"]
        log.info(
            f"  {name:<32} {r['fs_r2_train']:>6.3f} {r['fs_r2_val']:>6.3f} "
            f"{r['fs_r2_test']:>6.3f}  "
            f"{r['sindy_r2_test']:>6.3f} {r['delta_test']:>+7.3f}  "
            f"{r['fs_spec_test']:>8.3f}"
        )
    log.info(f"  {'─'*80}")
    log.info(
        f"  {'MEAN':<32} {mean_train:>6.3f} {mean_val:>6.3f} "
        f"{mean_test:>6.3f}  {mean_sindy:>6.3f} {mean_delta:>+7.3f}"
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    log.info(f"\n── Saving outputs ...")

    # Province breakdown
    pd.DataFrame(prov_rows).to_csv(
        os.path.join(RESULTS_DIR, "province_breakdown.csv"), index=False)

    # Epoch history
    epoch_rows = []
    for rec in coordinator.epoch_history:
        row = {
            "epoch"              : rec["epoch"],
            "global_best_fitness": rec["global_best_fitness"],
            "migrant_fitness"    : rec.get("migrant_fitness", float("nan")),
            "migrant_n_terms"    : rec.get("migrant_n_terms", 0),
        }
        row.update({f"island_{k}_fitness": v
                    for k, v in rec["island_fitnesses"].items()})
        epoch_rows.append(row)
    pd.DataFrame(epoch_rows).to_csv(
        os.path.join(RESULTS_DIR, "epoch_history.csv"), index=False)

    # Best equation
    pd.DataFrame([{
        "equation"       : str(best_eq),
        "n_terms"        : best_eq.n_terms(),
        "val_fitness"    : coordinator._global_best_fitness,
        "mean_r2_test"   : round(mean_test,  4),
        "mean_r2_sindy"  : round(mean_sindy, 4),
        "delta_test"     : round(mean_delta, 4),
        "n_provinces"    : len(province_dfs),
        "n_islands"      : N_ISLANDS,
    }]).to_csv(os.path.join(RESULTS_DIR, "best_equation.csv"), index=False)

    # Human-readable summary
    summary_path = os.path.join(RESULTS_DIR, "final_summary.txt")
    with open(summary_path, "w") as f:
        f.write("funsearch-ga-sindy — Final Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best equation ({best_eq.n_terms()} terms):\n")
        f.write(f"  {best_eq}\n\n")
        f.write(f"Provinces used       : {len(province_dfs)}\n")
        f.write(f"Islands              : {N_ISLANDS}\n")
        f.write(f"Mean test R² (GA)    : {mean_test:.4f}\n")
        f.write(f"Mean test R² (SINDy) : {mean_sindy:.4f}\n")
        f.write(f"Delta                : {mean_delta:+.4f}\n\n")
        f.write("Per-province results:\n")
        for r in prov_rows:
            f.write(f"  {r['province']:<32}  "
                    f"test={r['fs_r2_test']:.3f}  "
                    f"sindy={r['sindy_r2_test']:.3f}  "
                    f"delta={r['delta_test']:+.3f}\n")

    for fname in ["province_breakdown.csv", "epoch_history.csv",
                  "best_equation.csv", "final_summary.txt", "run_ga_sindy.log"]:
        log.info(f"  -> results/{fname}")

    elapsed = time.time() - t_start
    log.info(f"\n  Runtime: {elapsed:.1f}s  |  Done.")


if __name__ == "__main__":
    main()
