"""
island.py
─────────────────────────────────────────────────────────────────────────────
Island class for FunSearch-SINDy (Project 2).

Architecture
────────────
One Island = one biological grouping of provinces (Coastal or Inland).

Seed
  The initial equation structure is the UNION of all SINDy M1 term structures
  from every province in the island. Coefficients are not seeded — they are
  always re-fitted from data.

Fixed single split
  train : 2015–2019  (5 years, 260 rows/province)
  val   : 2020–2021  (2 years, used as fitness signal throughout evolution)
  test  : 2022–2023  (2 years, never touched until final evaluation)

Per-window loop (N_WINDOWS=1 with this split)
  For each generation:
    1. Tournament-select a parent from the population
    2. Mutate the structure (add / remove / swap / crossover terms)
    3. Re-fit coefficients via least squares on training data
    4. Evaluate fitness (mean R²) on both val years
    5. Replace weakest population member if the mutant improves fitness

Coefficient re-fitting (SINDy inner loop)
  Given a fixed set of terms T and training rows, solve:
      Θ(T) · ξ = ẋ      (ordinary least squares, no thresholding)
  This gives the optimal coefficients for that term structure and training
  data. Structural search is handled by the mutator; coefficient
  optimisation is handled here.

Public API
──────────
  Island(name, provinces, province_dfs, seed_terms)
      province_dfs : list of province DataFrames (full data, all years)
      seed_terms   : set[str]  — union of SINDy term names for this island

  island.run_all_windows(verbose=True) → List[Equation]
      Run all windows in sequence, return list of best equations
      (one per window; with single split this is a list of length 1).

  island.get_best() → Equation
      Return the best equation found so far.

  island.fitness_history  List[float]
      Best fitness at each generation.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Set, Dict, Optional
from dataclasses import dataclass, field

from config import (
    TARGET_COL,
    LIBRARY,
    WINDOWS, N_WINDOWS,
    POPULATION_SIZE,
    GENERATIONS_PER_WINDOW,
    TOURNAMENT_K,
    EARLY_STOP_DELTA,
    EARLY_STOP_PATIENCE,
    MAX_TERMS,
    MUTATION_PROBS,
    NEW_TERM_SIGMA,
)
from equation import Equation, _term_values
from evaluator import score_province, shape_score
from mutator import mutate


# ── Population member ─────────────────────────────────────────────────────────

@dataclass
class Member:
    equation : Equation
    fitness  : float
    origin   : str   # "seed" | "mutant" | "migrant"
    window   : int   # window index when this member was created
    age      : int = 0


# ── Island ────────────────────────────────────────────────────────────────────

class Island:
    """
    One evolutionary island managing a population of Equation objects.
    Runs 4 walk-forward windows; best equation carries forward between windows.
    """

    def __init__(
        self,
        name         : str,
        provinces    : List[str],
        province_dfs : List[pd.DataFrame],
        seed_terms   : Set[str],
    ):
        self.name         = name
        self.provinces    = provinces
        self.province_dfs = province_dfs   # one DataFrame per province, all years
        self.seed_terms   = seed_terms     # union of SINDy term structures

        # Sort province DFs by shape_score descending — high-signal provinces
        # carry more weight in fitness evaluation (mirrors evaluator.fitness())
        scores = [shape_score(df) for df in province_dfs]
        order  = np.argsort(scores)[::-1]
        self.province_dfs = [province_dfs[i] for i in order]
        self.provinces    = [provinces[i]    for i in order]

        self.population      : List[Member] = []
        self.best_equation   : Optional[Equation] = None
        self.best_fitness    : float = -np.inf
        self.fitness_history : List[float] = []   # best fitness per generation
        self._window_results : List[Equation] = []


    # ── Public entry point ────────────────────────────────────────────────────

    def run_all_windows(self, verbose: bool = True) -> List[Equation]:
        """
        Run all 4 walk-forward windows in sequence.

        Returns
        -------
        List[Equation]  — best equation for each window (length = N_WINDOWS).
        Window 4's equation is the primary deliverable.
        """
        carry_terms: Optional[Set[str]] = None   # None → use seed_terms for W1

        for w_idx, window in enumerate(WINDOWS):
            if verbose:
                print(f"\n{'─'*60}")
                print(f"  Island {self.name}  |  Window {w_idx+1}/{N_WINDOWS}")
                print(f"  Train: {window['train']}  →  Val: {window['val']}  →  Test: {window['test']}")
                print(f"{'─'*60}")

            # Initialise population for this window
            start_terms = carry_terms if carry_terms is not None else self.seed_terms
            self._initialise_population(start_terms, window, w_idx, verbose)

            # Run evolution
            best = self._run_window(window, w_idx, verbose)
            self._window_results.append(best)
            carry_terms = set(best.terms.keys())   # carry structure into next window

            if verbose:
                print(f"  Window {w_idx+1} best fitness : {self.best_fitness:.4f}")
                print(f"  Best equation : {best}")

        return self._window_results


    def get_best(self) -> Optional[Equation]:
        return self.best_equation


    # ── Initialisation ────────────────────────────────────────────────────────

    def _initialise_population(
        self,
        start_terms : Set[str],
        window      : dict,
        w_idx       : int,
        verbose     : bool,
    ) -> None:
        """
        Build a fresh population for one window.

        Member 0   : seed equation (start_terms, re-fitted coefficients)
        Members 1–N: structural mutations of the seed, each re-fitted
        """
        self.population = []

        # Build seed equation — all start_terms with placeholder coefficients
        # (coefficients will be overwritten by refit immediately)
        seed_coefs = {t: np.random.normal(0, NEW_TERM_SIGMA)
                      for t in start_terms if t in LIBRARY}

        # If seed is too large, trim to MAX_TERMS by random sampling
        if len(seed_coefs) > MAX_TERMS:
            kept = np.random.choice(list(seed_coefs.keys()), MAX_TERMS, replace=False)
            seed_coefs = {t: seed_coefs[t] for t in kept}

        seed_eq = self._refit(Equation(seed_coefs), window["train"])
        seed_f  = self._evaluate(seed_eq, window["val"], window["train"])
        self.population.append(Member(seed_eq, seed_f, "seed", w_idx))

        if verbose:
            print(f"  Seed: {len(seed_eq.terms)} terms, "
                  f"val R²={seed_f:.4f}  ({seed_eq})")

        # Fill remaining slots with structural mutations of the seed
        attempts = 0
        while len(self.population) < POPULATION_SIZE and attempts < POPULATION_SIZE * 10:
            attempts += 1
            try:
                # Pick a random existing member as parent
                parent = np.random.choice(self.population).equation
                mutant_struct = mutate(parent, self.province_dfs)
                mutant = self._refit(mutant_struct, window["train"])
                fit    = self._evaluate(mutant, window["val"], window["train"])
                self.population.append(Member(mutant, fit, "mutant", w_idx))
            except Exception:
                continue

        # Pad with seed copies if we couldn't fill
        while len(self.population) < POPULATION_SIZE:
            m = Member(seed_eq.copy(), seed_f, "seed", w_idx)
            self.population.append(m)

        # Track global best
        self._update_global_best()


    # ── Evolution loop ────────────────────────────────────────────────────────

    def _run_window(
        self,
        window  : dict,
        w_idx   : int,
        verbose : bool,
    ) -> Equation:
        """
        Run GENERATIONS_PER_WINDOW generations for one window.
        Returns the best equation found.
        """
        no_improve = 0
        prev_best  = self.best_fitness

        for gen in range(GENERATIONS_PER_WINDOW):

            # ── Tournament selection ──────────────────────────────────────────
            k        = min(TOURNAMENT_K, len(self.population))
            contenders = np.random.choice(len(self.population), k, replace=False)
            parent_idx = max(contenders, key=lambda i: self.population[i].fitness)
            parent     = self.population[parent_idx].equation

            # Pick second parent for crossover (different from parent)
            second = None
            if len(self.population) > 1:
                others = [i for i in range(len(self.population)) if i != parent_idx]
                second = self.population[np.random.choice(others)].equation

            # ── Mutate structure ──────────────────────────────────────────────
            try:
                mutant_struct = mutate(parent, self.province_dfs, second_parent=second)
            except Exception:
                continue

            # ── Re-fit coefficients via least squares ─────────────────────────
            try:
                mutant = self._refit(mutant_struct, window["train"])
            except Exception:
                continue

            if mutant.n_terms() == 0:
                continue

            # ── Evaluate on val year ──────────────────────────────────────────
            fit = self._evaluate(mutant, window["val"], window["train"])

            # ── Replace weakest if improvement ────────────────────────────────
            weakest_idx = min(range(len(self.population)),
                              key=lambda i: self.population[i].fitness)
            if fit > self.population[weakest_idx].fitness:
                self.population[weakest_idx] = Member(mutant, fit, "mutant", w_idx)

            # ── Track best ────────────────────────────────────────────────────
            self._update_global_best()
            self.fitness_history.append(self.best_fitness)

            # Age all members
            for m in self.population:
                m.age += 1

            # ── Early stopping ────────────────────────────────────────────────
            if self.best_fitness - prev_best >= EARLY_STOP_DELTA:
                no_improve = 0
                prev_best  = self.best_fitness
            else:
                no_improve += 1

            if no_improve >= EARLY_STOP_PATIENCE:
                if verbose:
                    print(f"    Early stop at gen {gen+1}  "
                          f"(no improvement for {EARLY_STOP_PATIENCE} gens)")
                break

            if verbose and (gen + 1) % 10 == 0:
                print(f"    Gen {gen+1:3d}  best={self.best_fitness:.4f}  "
                      f"terms={self.best_equation.n_terms()}")

        return self.best_equation.copy()


    # ── Coefficient re-fitting (SINDy inner loop) ─────────────────────────────

    def _refit(self, equation: Equation, train_years: List[int]) -> Equation:
        """
        Given a term structure (equation.terms.keys()), fit optimal coefficients
        via ordinary least squares on all training province data.

        Solves:  Θ(terms) · ξ = ẋ_train
        Returns a new Equation with the OLS-fitted coefficients.
        Falls back to the input equation if fitting fails.
        """
        terms = list(equation.terms.keys())
        if not terms:
            return equation.copy()

        X_blocks, y_blocks = [], []

        for df in self.province_dfs:
            subset = df[df["year"].isin(train_years)].copy()
            if len(subset) < max(4, len(terms)):
                continue   # too few rows to fit this many terms

            y = subset[TARGET_COL].values.astype(float)
            if not np.all(np.isfinite(y)):
                continue

            try:
                cols = []
                for t in terms:
                    v = _term_values(t, subset).astype(float)
                    if not np.all(np.isfinite(v)):
                        raise ValueError(f"Non-finite values in term {t}")
                    cols.append(v)
                X_blocks.append(np.column_stack(cols))
                y_blocks.append(y)
            except Exception:
                continue

        if not X_blocks:
            return equation.copy()   # no valid training data — keep old coefficients

        X = np.vstack(X_blocks)
        y = np.concatenate(y_blocks)

        # Clip target to guard against extreme outliers
        y = np.clip(y, -100.0, 100.0)

        try:
            coeffs, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            return equation.copy()

        if not np.all(np.isfinite(coeffs)):
            return equation.copy()

        # Drop terms whose fitted coefficient collapsed to near zero
        new_terms = {t: float(c) for t, c in zip(terms, coeffs)
                     if abs(c) > 1e-6}

        return Equation(new_terms) if new_terms else equation.copy()


    # ── Fitness evaluation ────────────────────────────────────────────────────

    def _evaluate(
        self,
        equation    : Equation,
        val_years,          # int or List[int]
        train_years : List[int],
    ) -> float:
        """
        Simple mean R² across all island provinces on val_years.
        Accepts either a single year (int) or a list of years.
        (No shape-weighting — every province counts equally.)
        """
        if equation.n_terms() == 0:
            return -1.0

        if isinstance(val_years, int):
            val_years = [val_years]

        scores = [score_province(equation, df, years=val_years)
                  for df in self.province_dfs]

        return float(np.mean(scores))


    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_global_best(self) -> None:
        """Update self.best_equation and self.best_fitness from the population."""
        for m in self.population:
            if m.fitness > self.best_fitness:
                self.best_fitness  = m.fitness
                self.best_equation = m.equation.copy()


    def summary(self) -> str:
        """One-line summary of island state."""
        n = len(self.population)
        fits = [m.fitness for m in self.population]
        return (f"Island {self.name:<20}  "
                f"pop={n}  "
                f"best={self.best_fitness:.4f}  "
                f"mean={np.mean(fits):.4f}  "
                f"terms={self.best_equation.n_terms() if self.best_equation else 0}")


# ── Seed builder (called by run_funsearch.py) ──────────────────────────────────

def build_seed_terms(
    province_list : List[str],
    eq_lookup     : Dict[str, str],
) -> Set[str]:
    """
    Compute the union of SINDy M1 term structures for all provinces in an island.

    Parameters
    ----------
    province_list : canonical province names for this island
    eq_lookup     : dict mapping province name → equation string (from m1_summary.csv)

    Returns
    -------
    set[str]  — canonical term names present in at least one province's equation
    """
    from equation import Equation

    all_terms: Set[str] = set()
    for prov in province_list:
        eq_str = eq_lookup.get(prov)
        if eq_str is None:
            num = prov.split()[0]
            eq_str = next((v for k, v in eq_lookup.items()
                           if k.startswith(num)), None)
        if eq_str:
            try:
                eq = Equation.parse(eq_str)
                all_terms |= set(eq.terms.keys())
            except Exception:
                pass

    return all_terms


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from config import SEIR_CSV, M1_SUMMARY, ISLANDS, TRAIN_YEARS, VAL_YEARS

    print("Loading data …")
    feat_df = pd.read_csv(SEIR_CSV).sort_values(["province", "year", "week"])
    m1      = pd.read_csv(M1_SUMMARY)
    eq_lookup = dict(zip(m1["province"].str.strip(), m1["equation"].str.strip()))

    # Build Island A (Coastal) only for the smoke test
    island_provs = ISLANDS[0]
    dfs = []
    for prov in island_provs:
        sub = feat_df[feat_df["province"] == prov]
        if len(sub) == 0:
            num = prov.split()[0]
            candidates = [p for p in feat_df["province"].unique()
                          if str(p).startswith(num)]
            if candidates:
                sub = feat_df[feat_df["province"] == candidates[0]]
        if len(sub) > 0:
            dfs.append(sub.copy())

    seed_terms = build_seed_terms(island_provs, eq_lookup)

    print(f"\nIsland A — {len(dfs)} provinces loaded")
    print(f"Seed terms ({len(seed_terms)}): {sorted(seed_terms)[:5]} ...")

    island = Island(
        name         = "A_coastal",
        provinces    = island_provs[:len(dfs)],
        province_dfs = dfs,
        seed_terms   = seed_terms,
    )

    print("\nRunning all windows …")
    results = island.run_all_windows(verbose=True)

    print("\n── Final results ──────────────────────────────────────────")
    for i, eq in enumerate(results):
        print(f"  Window {i+1}: {eq}")
    print(f"\n  Overall best fitness : {island.best_fitness:.4f}")
    print(f"  {island.summary()}")
