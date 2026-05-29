"""
island.py
─────────────────────────────────────────────────────────────────────────────
Single evolutionary island for funsearch-ga-sindy.

Search algorithm: Simulated Annealing (SA)
──────────────────────────────────────────
Each island runs SA to search the discrete space of equation structures
(which terms from the 66-term library to include).

At every generation:
  1. Apply one random structural mutation to the current equation.
  2. OLS-refit coefficients on all province training data.
  3. Compute validation fitness.
  4. Accept or reject via the Metropolis criterion:
       - Always accept improvements (delta > 0).
       - Accept deteriorations with probability exp(delta / T).
  5. Cool the temperature: T *= SA_COOLING_RATE.

This replaces the old population-based random trial-and-error with a
principled optimisation algorithm that can escape local optima early in
the search and converge reliably as temperature falls.

Public API
──────────
  Island(name, province_dfs, seed_terms)
  island.run_epoch(train_years, val_years, verbose) -> Equation
  island.champion() -> Equation
  island.champion_fitness() -> float
  island.inject_migrant(equation, train_years)
  island.fitness_history -> List[float]
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Set, Optional

from config import (
    TARGET_COL, LIBRARY, TRAIN_YEARS, VAL_YEARS,
    GENERATIONS_PER_EPOCH, EARLY_STOP_PATIENCE,
    MAX_TERMS, NEW_TERM_SIGMA,
    SA_T_INITIAL, SA_T_FINAL, SA_COOLING_RATE,
    R2_WEIGHT, SPECTRAL_WEIGHT, N_SPECTRAL_FREQS,
    MUTATION_PROBS as _MUTATION_PROBS,
)
from equation import Equation
from evaluator import score_province, spectral_score, shape_score
from sindy_fitter import refit
from llm_mutator import _add_term, _remove_term, _swap_feature, _perturb


# ── Mutation operator ─────────────────────────────────────────────────────────

def _mutate_op(equation: Equation) -> Equation:
    """Apply one randomly chosen structural mutation."""
    if equation.n_terms() >= MAX_TERMS:
        return _remove_term(equation)
    if equation.n_terms() == 0:
        return _add_term(equation)

    ops   = list(_MUTATION_PROBS.keys())
    probs = np.array(list(_MUTATION_PROBS.values()), dtype=float)
    probs /= probs.sum()
    op    = np.random.choice(ops, p=probs)

    if op == "add_term":
        return _add_term(equation)
    elif op == "remove_term":
        return _remove_term(equation)
    elif op == "swap_feature":
        return _swap_feature(equation)
    else:
        return _perturb(equation)


# ── Island ────────────────────────────────────────────────────────────────────

class Island:
    """
    One independent evolutionary island using Simulated Annealing.
    Maintains a single current SA state and tracks the best-ever equation.
    """

    def __init__(
        self,
        name         : str,
        province_dfs : List[pd.DataFrame],
        seed_terms   : Set[str],
    ):
        self.name         = name
        self.province_dfs = province_dfs
        self.seed_terms   = seed_terms

        self._current_eq      : Optional[Equation] = None
        self._current_fitness : float = -np.inf
        self._best_eq         : Optional[Equation] = None
        self._best_fitness    : float = -np.inf

        self.fitness_history : List[float] = []
        self._initialised = False

    # ── Public API ────────────────────────────────────────────────────────────

    def champion(self) -> Optional[Equation]:
        return self._best_eq

    def champion_fitness(self) -> float:
        return self._best_fitness

    def inject_migrant(
        self,
        equation    : Equation,
        train_years : List[int],
    ) -> None:
        """
        Reset the SA current state to the GA migrant equation.
        This gives each island a fresh, cross-island starting point
        for the next epoch.
        """
        fitted  = refit(equation.terms.keys(), self.province_dfs, train_years)
        fitness = self._evaluate(fitted, VAL_YEARS)
        self._current_eq      = fitted
        self._current_fitness = fitness
        if fitness > self._best_fitness:
            self._best_fitness = fitness
            self._best_eq      = fitted.copy()

    def run_epoch(
        self,
        train_years : List[int] = None,
        val_years   : List[int] = None,
        verbose     : bool = True,
    ) -> Equation:
        """
        Run GENERATIONS_PER_EPOCH SA steps.
        Returns the best-ever equation found so far.
        """
        if train_years is None:
            train_years = TRAIN_YEARS
        if val_years is None:
            val_years = VAL_YEARS

        if not self._initialised:
            self._initialise(train_years, val_years, verbose)

        T          = SA_T_INITIAL
        no_improve = 0
        prev_best  = self._best_fitness

        for gen in range(GENERATIONS_PER_EPOCH):

            # Step 1. Structural mutation
            try:
                candidate_struct = _mutate_op(self._current_eq)
            except Exception:
                T *= SA_COOLING_RATE
                continue

            # Step 2. OLS refit
            try:
                candidate = refit(
                    candidate_struct.terms.keys(),
                    self.province_dfs,
                    train_years,
                )
            except Exception:
                T *= SA_COOLING_RATE
                continue

            if candidate.n_terms() == 0:
                T *= SA_COOLING_RATE
                continue

            # Step 3. Validation fitness
            candidate_fit = self._evaluate(candidate, val_years)
            delta         = candidate_fit - self._current_fitness

            # Step 4. Metropolis acceptance
            if delta > 0 or np.random.random() < np.exp(delta / max(T, 1e-10)):
                self._current_eq      = candidate
                self._current_fitness = candidate_fit

            # Step 5. Track best-ever
            if self._current_fitness > self._best_fitness:
                self._best_fitness = self._current_fitness
                self._best_eq      = self._current_eq.copy()

            self.fitness_history.append(self._best_fitness)

            # Step 6. Cool temperature
            T *= SA_COOLING_RATE

            # Early stopping on best-ever plateau
            if self._best_fitness - prev_best >= 0.001:
                no_improve = 0
                prev_best  = self._best_fitness
            else:
                no_improve += 1

            if no_improve >= EARLY_STOP_PATIENCE:
                if verbose:
                    print(f"    [{self.name}] Early stop gen {gen+1}  "
                          f"T={T:.5f}")
                break

            if verbose and (gen + 1) % 20 == 0:
                print(f"    [{self.name}] gen {gen+1:3d}  "
                      f"T={T:.4f}  "
                      f"cur={self._current_fitness:.4f}  "
                      f"best={self._best_fitness:.4f}  "
                      f"terms={self._best_eq.n_terms()}")

        return self._best_eq.copy()

    # ── Fitness ───────────────────────────────────────────────────────────────

    def _evaluate(self, equation: Equation, years: List[int]) -> float:
        if equation.n_terms() == 0:
            return -1.0
        combined, weights = [], []
        for df in self.province_dfs:
            r2   = score_province(equation, df, years=years)
            spec = spectral_score(equation, df, years=years,
                                  n_freqs=N_SPECTRAL_FREQS)
            w    = shape_score(df)
            combined.append(R2_WEIGHT * r2 + SPECTRAL_WEIGHT * spec)
            weights.append(w)
        combined = np.array(combined, dtype=float)
        weights  = np.array(weights,  dtype=float)
        total_w  = weights.sum()
        if total_w < 1e-9:
            return float(np.mean(combined))
        return float(np.sum(combined * weights) / total_w)

    # ── Initialisation ────────────────────────────────────────────────────────

    def _initialise(
        self,
        train_years : List[int],
        val_years   : List[int],
        verbose     : bool,
    ) -> None:
        seed_coefs = {t: np.random.normal(0, NEW_TERM_SIGMA)
                      for t in self.seed_terms if t in LIBRARY}
        if len(seed_coefs) > MAX_TERMS:
            kept = list(np.random.choice(
                list(seed_coefs.keys()), MAX_TERMS, replace=False))
            seed_coefs = {t: seed_coefs[t] for t in kept}

        seed_eq  = refit(seed_coefs.keys(), self.province_dfs, train_years)
        seed_fit = self._evaluate(seed_eq, val_years)

        self._current_eq      = seed_eq
        self._current_fitness = seed_fit
        self._best_eq         = seed_eq.copy()
        self._best_fitness    = seed_fit
        self._initialised     = True

        if verbose:
            print(f"  [{self.name}] SA seed: {seed_eq.n_terms()} terms  "
                  f"val_fitness={seed_fit:.4f}  "
                  f"T_start={SA_T_INITIAL}  T_end={SA_T_FINAL}")

    def summary(self) -> str:
        return (f"Island {self.name:<12}  "
                f"best={self._best_fitness:.4f}  "
                f"cur={self._current_fitness:.4f}  "
                f"terms={self._best_eq.n_terms() if self._best_eq else 0}")
