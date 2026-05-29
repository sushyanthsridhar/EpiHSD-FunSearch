"""
seed_loader.py
─────────────────────────────────────────────────────────────────────────────
Bridge between Project 1 (SINDy) and Project 2 (FunSearch-SINDy).

Reads sindy/results/m1_summary.csv, parses each province's discovered
equation into an Equation object, loads the corresponding province data
from seir_features.csv, and returns two island bundles ready for the
evolution loop.

Public API
──────────
  load_islands() → (IslandBundle, IslandBundle)
      Returns (island_a, island_b).

  IslandBundle  — a dataclass with:
      .name         str             island label ("A_coastal" / "B_inland")
      .provinces    List[str]       province names in this island
      .dfs          List[DataFrame] province DataFrames (train+val+test)
      .train_dfs    List[DataFrame] training years only
      .val_dfs      List[DataFrame] validation years only
      .seeds        List[Equation]  SINDy best equations (one per province)
      .seed_fitness float           baseline fitness on val set
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import sys
import pandas as pd
from dataclasses import dataclass, field
from typing import List

from config import (
    SEIR_CSV, M1_SUMMARY,
    ISLANDS, ALL_PROVINCES,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
)
from equation import Equation
from evaluator import fitness as eval_fitness


@dataclass
class IslandBundle:
    name        : str
    provinces   : List[str]
    dfs         : List[pd.DataFrame]       # full data (all years)
    train_dfs   : List[pd.DataFrame]
    val_dfs     : List[pd.DataFrame]
    test_dfs    : List[pd.DataFrame]
    seeds       : List[Equation]           # one per province, from SINDy M1
    seed_fitness: float = 0.0             # baseline val fitness


# ── Main loader ───────────────────────────────────────────────────────────────

def load_islands(verbose: bool = True) -> tuple[IslandBundle, IslandBundle]:
    """
    Load SINDy seeds and province data, return two populated IslandBundles.
    """

    # ── Load seir_features.csv ────────────────────────────────────────────────
    print("Loading seir_features.csv …")
    feat_df = pd.read_csv(SEIR_CSV)
    feat_df = feat_df.sort_values(["province", "year", "week"]).reset_index(drop=True)
    provinces_in_data = set(feat_df["province"].unique())
    print(f"  {len(feat_df):,} rows  |  {len(provinces_in_data)} provinces")

    # ── Load m1_summary.csv ───────────────────────────────────────────────────
    print("Loading m1_summary.csv …")
    m1 = pd.read_csv(M1_SUMMARY)
    # Build province → equation string lookup
    eq_lookup = dict(zip(m1["province"].str.strip(), m1["equation"].str.strip()))
    print(f"  {len(eq_lookup)} province equations found")

    # ── Build each island ─────────────────────────────────────────────────────
    all_islands = []
    for i, province_list in enumerate(ISLANDS):
        name = province_list[0] if len(province_list) == 1 else f"island_{i}"
        all_islands.append(_build_island(name, province_list, feat_df, eq_lookup, verbose))

    return all_islands


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_island(
    name: str,
    province_list: List[str],
    feat_df: pd.DataFrame,
    eq_lookup: dict,
    verbose: bool,
) -> IslandBundle:

    dfs, train_dfs, val_dfs, test_dfs, seeds = [], [], [], [], []
    missing_data, missing_eq, parse_errors = [], [], []

    for prov in province_list:
        # ── Province data ─────────────────────────────────────────────────────
        sub = feat_df[feat_df["province"] == prov].copy()
        if len(sub) == 0:
            # Try fuzzy match on province number prefix
            num = prov.split()[0]
            candidates = [p for p in feat_df["province"].unique()
                          if str(p).startswith(num)]
            if candidates:
                sub = feat_df[feat_df["province"] == candidates[0]].copy()
            if len(sub) == 0:
                missing_data.append(prov)
                continue

        dfs.append(sub)
        train_dfs.append(sub[sub["year"].isin(TRAIN_YEARS)])
        val_dfs.append(sub[sub["year"].isin(VAL_YEARS)])
        test_dfs.append(sub[sub["year"].isin(TEST_YEARS)])

        # ── Seed equation ─────────────────────────────────────────────────────
        eq_str = eq_lookup.get(prov)
        if eq_str is None:
            # Try fuzzy match on province number
            num = prov.split()[0]
            eq_str = next((v for k, v in eq_lookup.items()
                           if k.startswith(num)), None)

        if eq_str is None:
            missing_eq.append(prov)
            seeds.append(Equation())           # empty equation as placeholder
            continue

        try:
            eq = Equation.parse(eq_str)
            seeds.append(eq)
        except Exception as e:
            parse_errors.append((prov, str(e)))
            seeds.append(Equation())

    # ── Baseline fitness ──────────────────────────────────────────────────────
    # Score the population-average of all seed equations
    # (Most provinces have unique seeds — we average R² over them)
    if val_dfs and seeds:
        per_prov_r2 = []
        for df, eq in zip(val_dfs, seeds):
            from evaluator import score_province
            per_prov_r2.append(score_province(eq, df.assign(
                **{c: df[c] for c in df.columns}
            )))
        import numpy as np
        baseline = float(np.mean(per_prov_r2))
    else:
        baseline = 0.0

    if verbose:
        print(f"\n  Island {name}:")
        print(f"    Provinces loaded    : {len(dfs)}")
        print(f"    Seed equations      : {len([s for s in seeds if s.n_terms() > 0])}")
        print(f"    Baseline val R²     : {baseline:.4f}")
        if missing_data:
            print(f"    WARN missing data   : {missing_data}")
        if missing_eq:
            print(f"    WARN missing eq     : {missing_eq}")
        if parse_errors:
            print(f"    WARN parse errors   : {parse_errors}")
        print(f"    Sample seed (first) : {seeds[0] if seeds else 'none'}")

    bundle = IslandBundle(
        name         = name,
        provinces    = province_list[:len(dfs)],
        dfs          = dfs,
        train_dfs    = train_dfs,
        val_dfs      = val_dfs,
        test_dfs     = test_dfs,
        seeds        = seeds,
        seed_fitness = baseline,
    )
    return bundle


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    a, b = load_islands(verbose=True)
    print(f"\n=== Island A: {a.name} ===")
    print(f"  Provinces : {len(a.provinces)}")
    print(f"  Seeds     : {a.seeds[0]}")
    print(f"  Val R²    : {a.seed_fitness:.4f}")

    print(f"\n=== Island B: {b.name} ===")
    print(f"  Provinces : {len(b.provinces)}")
    print(f"  Seeds     : {b.seeds[0]}")
    print(f"  Val R²    : {b.seed_fitness:.4f}")
