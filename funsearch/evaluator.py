"""
evaluator.py
─────────────────────────────────────────────────────────────────────────────
Fitness evaluation for FunSearch-SINDy.

Public functions:

  score_province(equation, df, years)
      → R² on specified years for a single province.

  spectral_score(equation, df, years, n_freqs=5)
      → Spectral similarity at the first 5 non-DC frequencies.
        Measures whether the model captures epidemic periodicity,
        not just variance. Uses the full time series (train+val).

  fitness(equation, province_dfs, years)
      → Combined score: 0.7 * R² + 0.3 * spectral, shape-weighted
        across all provinces in an island. This is what the
        evolution loop maximises.

All evaluation is done on VAL_YEARS (2022) by default during evolution.
Final holdout evaluation on TEST_YEARS (2023) is done only in analysis.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List

from config import TARGET_COL, VAL_YEARS
from equation import Equation


# ── Province-level R² ─────────────────────────────────────────────────────────

def score_province(
    equation: Equation,
    df: pd.DataFrame,
    years: List[int] = None,
) -> float:
    """
    Evaluate the equation on one province and return R².

    Parameters
    ----------
    equation    : Equation object to evaluate
    df          : province DataFrame (must contain TARGET_COL and 'year')
    years       : which years to score on (defaults to VAL_YEARS from config)

    Returns
    -------
    float  — R² clamped to [-1.0, 1.0] to prevent extreme negatives
             from dominating the fitness landscape.
             Returns -1.0 on any numerical failure.
    """
    if years is None:
        years = VAL_YEARS

    subset = df[df["year"].isin(years)].copy()
    if len(subset) < 4:
        return -1.0  # too few points to score

    y_true = subset[TARGET_COL].values.astype(float)

    try:
        y_pred = equation.evaluate(subset)
    except Exception:
        return -1.0

    # Guard against NaN / Inf in predictions
    if not np.all(np.isfinite(y_pred)):
        return -1.0

    # Clip predictions to prevent runaway coefficients from dominating
    y_pred = np.clip(y_pred, -1000.0, 1000.0)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot < 1e-12:
        return -1.0  # target is constant — R² undefined

    r2 = 1.0 - ss_res / ss_tot
    return float(np.clip(r2, -1.0, 1.0))


# ── Spectral score ────────────────────────────────────────────────────────────

def spectral_score(
    equation: Equation,
    df: pd.DataFrame,
    years: List[int] = None,
    n_freqs: int = 5,
) -> float:
    """
    Measure how well the equation reproduces the power spectrum of the
    actual dI/dt signal at the first n_freqs non-DC frequencies.

    Why: R² rewards matching the amplitude of each week individually.
    Spectral score rewards matching the *periodic structure* — annual
    cycles, biannual peaks — which is what makes an epidemic model
    biologically meaningful.

    For a 52-week year, the first 5 non-DC frequencies correspond to
    periods of: 52, 26, 17.3, 13, and 10.4 weeks.

    Returns
    -------
    float in [0, 1].  1.0 = perfect spectral match.  0.0 = no match.
    Returns 0.0 on any numerical failure.
    """
    if years is None:
        years = VAL_YEARS

    subset = df[df["year"].isin(years)].copy()
    if len(subset) < 2 * n_freqs:
        return 0.0

    y_true = subset[TARGET_COL].values.astype(float)

    try:
        y_pred = equation.evaluate(subset)
    except Exception:
        return 0.0

    if not np.all(np.isfinite(y_pred)):
        return 0.0

    y_pred = np.clip(y_pred, -1000.0, 1000.0)

    # ── Compute power spectra ─────────────────────────────────────────────────
    # Use rfft (real FFT) — output length is N//2 + 1
    # Index 0 is DC (mean), indices 1..n_freqs are the first n_freqs harmonics
    psd_true = np.abs(np.fft.rfft(y_true - np.mean(y_true))) ** 2
    psd_pred = np.abs(np.fft.rfft(y_pred - np.mean(y_pred))) ** 2

    # Extract first n_freqs non-DC components (indices 1 to n_freqs inclusive)
    freqs_true = psd_true[1 : n_freqs + 1]
    freqs_pred = psd_pred[1 : n_freqs + 1]

    # Normalise each spectrum to sum to 1 (compare shape, not magnitude)
    norm_true = freqs_true / (freqs_true.sum() + 1e-12)
    norm_pred = freqs_pred / (freqs_pred.sum() + 1e-12)

    # Score: 1 minus mean absolute relative error across the 5 frequencies
    # Clamped to [0, 1]
    mae = np.mean(np.abs(norm_true - norm_pred))
    score = float(np.clip(1.0 - mae, 0.0, 1.0))
    return score


# ── Shape score (epidemic peak sharpness weight) ──────────────────────────────

def shape_score(df: pd.DataFrame) -> float:
    """
    Weight for a province: max(incidence) / mean(incidence).
    Provinces with sharper epidemic peaks contribute more to fitness.
    A flat signal (endemic plateau) gets weight ≈ 1.
    """
    inc = df["incidence"].values.astype(float)
    mean_inc = np.mean(inc)
    if mean_inc < 1e-9:
        return 1.0
    return float(np.max(inc) / mean_inc)


# ── Fitness weights ───────────────────────────────────────────────────────────
R2_WEIGHT       = 0.7   # weight on R² component
SPECTRAL_WEIGHT = 0.3   # weight on spectral similarity component
N_SPECTRAL_FREQS = 5    # number of non-DC frequencies to compare


# ── Island-level fitness ──────────────────────────────────────────────────────

def fitness(
    equation: Equation,
    province_dfs: List[pd.DataFrame],
    years: List[int] = None,
) -> float:
    """
    Combined shape-weighted fitness across all provinces in an island.

    fitness = 0.7 * R²  +  0.3 * spectral_similarity

    R² rewards matching the week-by-week amplitude of dI/dt.
    Spectral similarity rewards capturing the epidemic periodicity
    at the 5 dominant frequencies (annual, biannual, and harmonics).

    Parameters
    ----------
    equation       : Equation to evaluate
    province_dfs   : list of province DataFrames for one island
    years          : scoring years (default VAL_YEARS)

    Returns
    -------
    float — combined weighted score.  Higher is better.
    """
    if years is None:
        years = VAL_YEARS

    if not province_dfs:
        return -1.0

    combined = []
    weights  = []

    for df in province_dfs:
        r2   = score_province(equation, df, years=years)
        spec = spectral_score(equation, df, years=years, n_freqs=N_SPECTRAL_FREQS)
        w    = shape_score(df)

        score = R2_WEIGHT * r2 + SPECTRAL_WEIGHT * spec
        combined.append(score)
        weights.append(w)

    combined = np.array(combined, dtype=float)
    weights  = np.array(weights,  dtype=float)

    total_weight = weights.sum()
    if total_weight < 1e-9:
        return float(np.mean(combined))

    return float(np.sum(combined * weights) / total_weight)


# ── Convenience: score on multiple year splits ────────────────────────────────

def score_splits(
    equation: Equation,
    province_dfs: List[pd.DataFrame],
) -> dict:
    """
    Return fitness on train / val / test splits in one call.
    Useful for final reporting.
    """
    from config import TRAIN_YEARS, VAL_YEARS, TEST_YEARS
    return {
        "r2_train" : fitness(equation, province_dfs, years=TRAIN_YEARS),
        "r2_val"   : fitness(equation, province_dfs, years=VAL_YEARS),
        "r2_test"  : fitness(equation, province_dfs, years=TEST_YEARS),
    }
