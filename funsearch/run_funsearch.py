"""
run_funsearch.py
─────────────────────────────────────────────────────────────────────────────
Main entry point for FunSearch-SINDy (Project 2).

Runs both islands (Coastal A, Inland B) through the fixed train/val/test
split and prints a full comparison against the SINDy M1 baseline.

  Train : 2015–2019  (5 years, 260 rows/province)
  Val   : 2020–2021  (2 years, fitness signal)
  Test  : 2022–2023  (2 years, held-out)

Usage
─────
    cd funsearch/
    python run_funsearch.py

Outputs  (all written to funsearch/results/)
────────
    best_equations.csv        — best equation per island per window
    fitness_history.csv       — best fitness at every generation
    province_breakdown.csv    — per-province R² + spectral score (train/val/test)
    final_summary.csv         — Island A/B final equations vs SINDy baseline
    run_funsearch.log         — full console log
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
    ISLANDS, WINDOWS,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
)
from equation import Equation
from evaluator import score_province, spectral_score
from island import Island, build_seed_terms

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(RESULTS_DIR, "run_funsearch.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="w"),
    ],
)
log = logging.getLogger(__name__)

ISLAND_NAMES = ["A_coastal", "B_inland"]


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    log.info("=" * 68)
    log.info("  FunSearch-SINDy — Project 2")
    log.info("  DR Dengue · Fixed-Split Island Evolution (train 2015-19 | val 2020-21 | test 2022-23)")
    log.info("=" * 68)

    log.info("\n── Loading seir_features.csv …")
    feat_df = (pd.read_csv(SEIR_CSV)
               .sort_values(["province", "year", "week"])
               .reset_index(drop=True))
    log.info(f"   {len(feat_df):,} rows  |  {feat_df['province'].nunique()} provinces")
    log.info(f"   Split: train={TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}  "
             f"val={VAL_YEARS[0]}–{VAL_YEARS[-1]}  "
             f"test={TEST_YEARS[0]}–{TEST_YEARS[-1]}")

    log.info("── Loading m1_summary.csv (SINDy M1 baseline) …")
    m1 = pd.read_csv(M1_SUMMARY)
    eq_lookup  = dict(zip(m1["province"].str.strip(), m1["equation"].str.strip()))
    sindy_test = dict(zip(m1["province"].str.strip(), m1["r2_test"]))
    sindy_val  = dict(zip(m1["province"].str.strip(), m1["r2_val"]))
    sindy_train= dict(zip(m1["province"].str.strip(), m1["r2_train"]))
    log.info(f"   {len(eq_lookup)} province equations")
    log.info(f"   SINDy mean test R² (all 32 provinces): {m1['r2_test'].mean():.4f}")

    return feat_df, eq_lookup, sindy_test, sindy_val, sindy_train


def build_province_dfs(feat_df, province_list):
    dfs, names, missing = [], [], []
    for prov in province_list:
        sub = feat_df[feat_df["province"] == prov]
        if len(sub) == 0:
            num = prov.split()[0]
            cands = [p for p in feat_df["province"].unique() if str(p).startswith(num)]
            sub = feat_df[feat_df["province"] == cands[0]] if cands else pd.DataFrame()
        if len(sub) > 0:
            dfs.append(sub.copy())
            names.append(prov)
        else:
            missing.append(prov)
    if missing:
        log.warning(f"   WARN: no data for {missing}")
    return dfs, names


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers  (simple mean — no shape weighting)
# ══════════════════════════════════════════════════════════════════════════════

def mean_r2(equation, province_dfs, years):
    """Simple mean R² across all provinces for the given years."""
    return float(np.mean([score_province(equation, df, years=years)
                          for df in province_dfs]))

def mean_spectral(equation, province_dfs, years):
    """Simple mean spectral score across all provinces for the given years."""
    return float(np.mean([spectral_score(equation, df, years=years)
                          for df in province_dfs]))

def sindy_mean(lookup, province_names):
    """Simple mean of SINDy R² values for the given provinces."""
    vals = [lookup.get(p, np.nan) for p in province_names]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


# ══════════════════════════════════════════════════════════════════════════════
# Per-province detailed breakdown
# ══════════════════════════════════════════════════════════════════════════════

def province_breakdown(equation, province_dfs, province_names,
                       sindy_test, sindy_val, sindy_train, island_name):
    """
    For every province in the island, compute R² and spectral score
    on train / val / test splits. Returns a list of dicts.
    """
    rows = []
    for prov, df in zip(province_names, province_dfs):
        r2_tr = score_province(equation, df, years=TRAIN_YEARS)
        r2_v  = score_province(equation, df, years=VAL_YEARS)
        r2_te = score_province(equation, df, years=TEST_YEARS)
        sp_tr = spectral_score(equation, df, years=TRAIN_YEARS)
        sp_v  = spectral_score(equation, df, years=VAL_YEARS)
        sp_te = spectral_score(equation, df, years=TEST_YEARS)
        rows.append({
            "island"         : island_name,
            "province"       : prov,
            "fs_r2_train"    : round(r2_tr, 4),
            "fs_r2_val"      : round(r2_v,  4),
            "fs_r2_test"     : round(r2_te, 4),
            "fs_spec_train"  : round(sp_tr, 4),
            "fs_spec_val"    : round(sp_v,  4),
            "fs_spec_test"   : round(sp_te, 4),
            "sindy_r2_train" : round(sindy_train.get(prov, np.nan), 4),
            "sindy_r2_val"   : round(sindy_val.get(prov, np.nan),   4),
            "sindy_r2_test"  : round(sindy_test.get(prov, np.nan),  4),
            "delta_test"     : round(r2_te - sindy_test.get(prov, np.nan), 4),
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    feat_df, eq_lookup, sindy_test, sindy_val, sindy_train = load_data()

    all_best_rows   = []
    all_history     = []
    all_province_rows = []
    summary_rows    = []

    for province_list, iname in zip(ISLANDS, ISLAND_NAMES):

        log.info(f"\n{'═'*68}")
        log.info(f"  ISLAND {iname}  ({len(province_list)} provinces)")
        log.info(f"{'═'*68}")

        province_dfs, loaded_provs = build_province_dfs(feat_df, province_list)

        seed_terms = build_seed_terms(province_list, eq_lookup)
        log.info(f"  Seed terms (union of SINDy M1): {len(seed_terms)}")

        base_test  = sindy_mean(sindy_test,  loaded_provs)
        base_val   = sindy_mean(sindy_val,   loaded_provs)
        base_train = sindy_mean(sindy_train, loaded_provs)
        log.info(f"  SINDy baseline  train={base_train:.4f}  val={base_val:.4f}  test={base_test:.4f}")

        island = Island(
            name         = iname,
            provinces    = loaded_provs,
            province_dfs = province_dfs,
            seed_terms   = seed_terms,
        )

        window_results = island.run_all_windows(verbose=True)

        # ── Per-window records ─────────────────────────────────────────────────
        for w_idx, (eq, window) in enumerate(zip(window_results, WINDOWS)):
            val_yrs  = window["val"]  if isinstance(window["val"],  list) else [window["val"]]
            test_yrs = window["test"] if isinstance(window["test"], list) else [window["test"]]
            all_best_rows.append({
                "island"    : iname,
                "window"    : w_idx + 1,
                "val_years" : str(val_yrs),
                "test_years": str(test_yrs),
                "val_r2"    : round(mean_r2(eq, province_dfs, val_yrs),  4),
                "test_r2"   : round(mean_r2(eq, province_dfs, test_yrs), 4),
                "n_terms"   : eq.n_terms(),
                "equation"  : str(eq),
            })

        # ── Fitness history ────────────────────────────────────────────────────
        for gen, fit in enumerate(island.fitness_history):
            all_history.append({"island": iname, "generation": gen+1, "best_fitness": fit})

        # ── Final equation — full evaluation ──────────────────────────────────
        final_eq = window_results[-1]

        r2_train  = mean_r2(final_eq,      province_dfs, TRAIN_YEARS)
        r2_val    = mean_r2(final_eq,      province_dfs, VAL_YEARS)
        r2_test   = mean_r2(final_eq,      province_dfs, TEST_YEARS)
        sp_train  = mean_spectral(final_eq, province_dfs, TRAIN_YEARS)
        sp_val    = mean_spectral(final_eq, province_dfs, VAL_YEARS)
        sp_test   = mean_spectral(final_eq, province_dfs, TEST_YEARS)

        delta_test = r2_test - base_test

        # ── Print final equation summary ───────────────────────────────────────
        tr_label  = f"R²  train ({TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]})"
        val_label = f"R²  val   ({VAL_YEARS[0]}–{VAL_YEARS[-1]})"
        te_label  = f"R²  test  ({TEST_YEARS[0]}–{TEST_YEARS[-1]})"
        sv_label  = f"Spectral  val  ({VAL_YEARS[0]}–{VAL_YEARS[-1]})"
        st_label  = f"Spectral  test ({TEST_YEARS[0]}–{TEST_YEARS[-1]})"
        log.info(f"\n  ── Final equation {'─'*48}")
        log.info(f"  {final_eq}")
        log.info(f"\n  {'Metric':<28} {'FunSearch':>10} {'SINDy M1':>10} {'Δ':>8}")
        log.info(f"  {'─'*58}")
        log.info(f"  {tr_label:<28} {r2_train:>10.4f} {base_train:>10.4f} {r2_train-base_train:>+8.4f}")
        log.info(f"  {val_label:<28} {r2_val:>10.4f} {base_val:>10.4f} {r2_val-base_val:>+8.4f}")
        log.info(f"  {te_label:<28} {r2_test:>10.4f} {base_test:>10.4f} {delta_test:>+8.4f}")
        log.info(f"  {'Spectral  train':<28} {sp_train:>10.4f} {'—':>10}")
        log.info(f"  {sv_label:<28} {sp_val:>10.4f} {'—':>10}")
        log.info(f"  {st_label:<28} {sp_test:>10.4f} {'—':>10}")

        # ── Per-province breakdown ─────────────────────────────────────────────
        prov_rows = province_breakdown(
            final_eq, province_dfs, loaded_provs,
            sindy_test, sindy_val, sindy_train, iname
        )
        all_province_rows.extend(prov_rows)

        log.info(f"\n  ── Per-province R² breakdown {'─'*38}")
        log.info(f"  {'Province':<28} {'FS_tr':>7} {'FS_v':>7} {'FS_te':>7} "
                 f"{'S_tr':>7} {'S_v':>7} {'S_te':>7} {'Δtest':>7}  {'Spec_te':>8}")
        log.info(f"  {'─'*95}")
        for r in prov_rows:
            name = r["province"].split(None, 1)[1] if " " in r["province"] else r["province"]
            log.info(
                f"  {name:<28} "
                f"{r['fs_r2_train']:>7.3f} {r['fs_r2_val']:>7.3f} {r['fs_r2_test']:>7.3f} "
                f"{r['sindy_r2_train']:>7.3f} {r['sindy_r2_val']:>7.3f} {r['sindy_r2_test']:>7.3f} "
                f"{r['delta_test']:>+7.3f}  {r['fs_spec_test']:>8.3f}"
            )
        log.info(f"  {'─'*95}")
        log.info(
            f"  {'MEAN':<28} "
            f"{np.mean([r['fs_r2_train'] for r in prov_rows]):>7.3f} "
            f"{np.mean([r['fs_r2_val']   for r in prov_rows]):>7.3f} "
            f"{np.mean([r['fs_r2_test']  for r in prov_rows]):>7.3f} "
            f"{np.mean([r['sindy_r2_train'] for r in prov_rows]):>7.3f} "
            f"{np.mean([r['sindy_r2_val']   for r in prov_rows]):>7.3f} "
            f"{np.mean([r['sindy_r2_test']  for r in prov_rows]):>7.3f} "
            f"{np.mean([r['delta_test']     for r in prov_rows]):>+7.3f}  "
            f"{np.mean([r['fs_spec_test']   for r in prov_rows]):>8.3f}"
        )

        summary_rows.append({
            "island"         : iname,
            "n_provinces"    : len(province_dfs),
            "n_terms"        : final_eq.n_terms(),
            "fs_r2_train"    : round(r2_train, 4),
            "fs_r2_val"      : round(r2_val,   4),
            "fs_r2_test"     : round(r2_test,  4),
            "fs_spec_train"  : round(sp_train, 4),
            "fs_spec_val"    : round(sp_val,   4),
            "fs_spec_test"   : round(sp_test,  4),
            "sindy_r2_train" : round(base_train, 4),
            "sindy_r2_val"   : round(base_val,   4),
            "sindy_r2_test"  : round(base_test,  4),
            "delta_test"     : round(delta_test, 4),
            "equation"       : str(final_eq),
        })

    # ── Save all outputs ──────────────────────────────────────────────────────
    log.info(f"\n{'═'*68}")
    log.info("  Saving outputs …")

    pd.DataFrame(all_best_rows).to_csv(
        os.path.join(RESULTS_DIR, "best_equations.csv"), index=False)

    pd.DataFrame(all_history).to_csv(
        os.path.join(RESULTS_DIR, "fitness_history.csv"), index=False)

    pd.DataFrame(all_province_rows).to_csv(
        os.path.join(RESULTS_DIR, "province_breakdown.csv"), index=False)

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(RESULTS_DIR, "final_summary.csv"), index=False)

    for f in ["best_equations.csv","fitness_history.csv",
              "province_breakdown.csv","final_summary.csv","run_funsearch.log"]:
        log.info(f"  → funsearch/results/{f}")

    # ── Master summary ────────────────────────────────────────────────────────
    log.info(f"\n{'═'*68}")
    log.info("  MASTER SUMMARY  (simple mean across provinces)")
    log.info(f"{'═'*68}")
    log.info(f"  {'Island':<14} {'Terms':>5}  "
             f"{'FS_train':>9} {'FS_val':>8} {'FS_test':>8}  "
             f"{'Sp_val':>7} {'Sp_test':>8}  "
             f"{'S_test':>8} {'Δ':>7}")
    log.info(f"  {'─'*90}")
    for r in summary_rows:
        arrow = "▲" if r["delta_test"] > 0 else "▼"
        log.info(
            f"  {r['island']:<14} {r['n_terms']:>5}  "
            f"{r['fs_r2_train']:>9.4f} {r['fs_r2_val']:>8.4f} {r['fs_r2_test']:>8.4f}  "
            f"{r['fs_spec_val']:>7.4f} {r['fs_spec_test']:>8.4f}  "
            f"{r['sindy_r2_test']:>8.4f} {r['delta_test']:>+7.4f} {arrow}"
        )

    elapsed = time.time() - t_start
    log.info(f"\n  Runtime: {elapsed:.1f}s  |  Done.")


if __name__ == "__main__":
    main()
