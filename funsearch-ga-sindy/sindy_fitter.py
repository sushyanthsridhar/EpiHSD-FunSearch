"""
sindy_fitter.py
─────────────────────────────────────────────────────────────────────────────
Standalone OLS coefficient fitter for the funsearch-ga-sindy pipeline.

Given a set of term names and a list of province DataFrames, solves:

    Theta(terms) . xi = dI_dt_norm_train

via ordinary least squares (no thresholding).  Returns a new Equation
with the fitted coefficients.

This is extracted from the old island.py _refit() method so it can be
reused by the GA coordinator, the island evolution loop, and analysis
scripts without importing the full Island class.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Iterable

from config import TARGET_COL, TRAIN_YEARS
from equation import Equation, _term_values


def refit(
    term_keys     : Iterable[str],
    province_dfs  : List[pd.DataFrame],
    train_years   : List[int] = None,
) -> Equation:
    """
    OLS-fit coefficients for the given term structure pooled across
    all supplied province DataFrames on the specified training years.

    Data is assembled in chronological province order:
        Province 0, week 1 ... Province 0, week T
        Province 1, week 1 ... Province 1, week T
        ...
    This respects the chronological ordering requested for the GA system.

    Parameters
    ----------
    term_keys     : iterable of library term name strings
    province_dfs  : list of pd.DataFrame (full data, all years, all columns)
    train_years   : list of int years to use for fitting (default TRAIN_YEARS)

    Returns
    -------
    Equation with OLS-fitted coefficients.
    Returns an Equation with zero coefficients on any numerical failure.
    """
    if train_years is None:
        train_years = TRAIN_YEARS

    terms = list(term_keys)
    if not terms:
        return Equation({})

    X_blocks : List[np.ndarray] = []
    y_blocks : List[np.ndarray] = []

    # Assemble design matrix province by province in order (chronological)
    for df in province_dfs:
        subset = df[df["year"].isin(train_years)].sort_values(
            ["year", "week"]).copy()

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

    new_terms = {t: float(c) for t, c in zip(terms, coeffs)
                 if abs(c) > 1e-6}
    return Equation(new_terms) if new_terms else Equation({t: 0.0 for t in terms})


def data_budget_report(province_dfs: List[pd.DataFrame],
                       train_years: List[int] = None) -> dict:
    """
    Print and return a summary of the available training data budget.
    Useful for computing the maximum sensible number of islands.
    """
    if train_years is None:
        train_years = TRAIN_YEARS

    total_rows  = 0
    prov_counts = []
    for df in province_dfs:
        sub = df[df["year"].isin(train_years)]
        total_rows += len(sub)
        prov_counts.append(len(sub))

    report = {
        "n_provinces"    : len(province_dfs),
        "total_train_rows": total_rows,
        "mean_rows_per_province": np.mean(prov_counts) if prov_counts else 0,
        "min_rows_per_province" : np.min(prov_counts)  if prov_counts else 0,
    }
    return report
