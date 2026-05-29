"""
ga_coordinator.py
─────────────────────────────────────────────────────────────────────────────
GA coordinator for funsearch-ga-sindy.

Sits above all islands and runs the inter-island genetic algorithm.

Each epoch proceeds as follows:

  1. Every island runs GENERATIONS_PER_EPOCH generations independently.

  2. GA coordinator collects one champion equation from each island
     (the best equation each island found this epoch).

  3. Tournament selection: pick the top GA_TOURNAMENT_K champions by
     their combined fitness.

  4. Crossover the top two selected champions to produce an offspring
     equation structure.

  5. Send the offspring to the LLM mutator to generate N_LLM_SUGGESTIONS
     structurally diverse variants.

  6. Refit each LLM variant via OLS on the training data.

  7. Evaluate each variant; keep the best one as the "GA migrant".

  8. Inject the GA migrant into ALL islands (replacing each island's
     weakest member, regardless of whether it improves — we always
     want cross-island gene flow).

  9. Record the global best equation across all islands.

 10. Repeat for MAX_EPOCHS epochs.

Public API
──────────
  GACoordinator(islands, province_dfs)
      islands      : list of Island objects (pre-constructed, not yet run)
      province_dfs : list of province DataFrames (same for all islands)

  coordinator.run(verbose) -> Equation
      Run the full evolution.  Returns the global best equation.

  coordinator.global_best() -> Equation
  coordinator.epoch_history -> List[dict]
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from config import (
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    MAX_EPOCHS, GA_TOURNAMENT_K, N_LLM_SUGGESTIONS,
    R2_WEIGHT, SPECTRAL_WEIGHT, N_SPECTRAL_FREQS,
)
from equation import Equation
from evaluator import score_province, spectral_score, shape_score
from sindy_fitter import refit
from llm_mutator import llm_mutate, _crossover
from island import Island


# ══════════════════════════════════════════════════════════════════════════════
# GA Coordinator
# ══════════════════════════════════════════════════════════════════════════════

class GACoordinator:
    """
    Orchestrates multiple Island objects via cross-island GA.
    """

    def __init__(
        self,
        islands      : List[Island],
        province_dfs : List[pd.DataFrame],
    ):
        self.islands      = islands
        self.province_dfs = province_dfs

        self._global_best_eq      : Optional[Equation] = None
        self._global_best_fitness : float = -np.inf
        self.epoch_history        : List[Dict[str, Any]] = []
        self._migrant_history     : List[Equation] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def global_best(self) -> Optional[Equation]:
        return self._global_best_eq

    def run(self, verbose: bool = True, max_epochs: int = None) -> Equation:
        """
        Run max_epochs epochs of island evolution + GA coordination.
        Defaults to MAX_EPOCHS from config if not specified.
        Returns the global best equation found.
        """
        n_epochs = max_epochs if max_epochs is not None else MAX_EPOCHS
        print(f"\n  GA Coordinator: {len(self.islands)} islands, "
              f"{n_epochs} epochs")
        print(f"  Data: {len(self.province_dfs)} quality-passing provinces")
        print("  " + "─" * 60)

        for epoch in range(n_epochs):
            if verbose:
                print(f"\n{'─'*64}")
                print(f"  EPOCH {epoch+1}/{MAX_EPOCHS}")
                print(f"{'─'*64}")

            # ── Step 1: Run each island for one epoch ─────────────────────────
            champions = []
            for island in self.islands:
                champion = island.run_epoch(
                    train_years=TRAIN_YEARS,
                    val_years=VAL_YEARS,
                    verbose=verbose,
                )
                fitness = island.champion_fitness()
                champions.append((island, champion, fitness))

                if verbose:
                    print(f"  {island.summary()}")

            # ── Step 2: Update global best ────────────────────────────────────
            for _, eq, fit in champions:
                if fit > self._global_best_fitness:
                    self._global_best_fitness = fit
                    self._global_best_eq      = eq.copy()

            # ── Step 3: GA tournament across island champions ─────────────────
            if verbose:
                print(f"\n  GA step — global best fitness: "
                      f"{self._global_best_fitness:.4f}  "
                      f"({self._global_best_eq.n_terms()} terms)")

            parent_a, parent_b = self._tournament_select(champions)

            # ── Step 4: Crossover ─────────────────────────────────────────────
            offspring = _crossover(parent_a, parent_b)
            if verbose:
                print(f"  Crossover: {parent_a.n_terms()} + "
                      f"{parent_b.n_terms()} terms -> "
                      f"{offspring.n_terms()} term offspring")

            # ── Step 5-6: LLM mutation + OLS refit ───────────────────────────
            if verbose:
                print(f"  Calling LLM mutator ({N_LLM_SUGGESTIONS} variants) ...")

            variants_struct = llm_mutate(
                equation      = offspring,
                province_dfs  = self.province_dfs,
                train_years   = TRAIN_YEARS,
                n_suggestions = N_LLM_SUGGESTIONS,
            )

            variants_fitted = []
            for v in variants_struct:
                try:
                    fitted = refit(v.terms.keys(), self.province_dfs, TRAIN_YEARS)
                    if fitted.n_terms() > 0:
                        variants_fitted.append(fitted)
                except Exception:
                    continue

            # Also include the offspring itself
            try:
                offspring_fitted = refit(offspring.terms.keys(),
                                         self.province_dfs, TRAIN_YEARS)
                if offspring_fitted.n_terms() > 0:
                    variants_fitted.append(offspring_fitted)
            except Exception:
                pass

            if not variants_fitted:
                if verbose:
                    print("  No valid LLM variants. Skipping migration.")
                self._record_epoch(epoch, champions, None, -np.inf)
                continue

            # ── Step 7: Select best variant as migrant ────────────────────────
            best_migrant    = None
            best_migrant_fit = -np.inf
            for v in variants_fitted:
                fit = self._eval_fitness(v, VAL_YEARS)
                if fit > best_migrant_fit:
                    best_migrant_fit = fit
                    best_migrant     = v.copy()

            if verbose:
                print(f"  Best migrant: {best_migrant.n_terms()} terms  "
                      f"val_fitness={best_migrant_fit:.4f}")

            # Update global best with migrant if it improves
            if best_migrant_fit > self._global_best_fitness:
                self._global_best_fitness = best_migrant_fit
                self._global_best_eq      = best_migrant.copy()
                if verbose:
                    print(f"  ** New global best from migrant: "
                          f"{self._global_best_fitness:.4f} **")

            # ── Step 8: Inject migrant into all islands ───────────────────────
            for island in self.islands:
                island.inject_migrant(best_migrant, TRAIN_YEARS)

            self._migrant_history.append(best_migrant.copy())
            self._record_epoch(epoch, champions, best_migrant, best_migrant_fit)

        print(f"\n{'='*64}")
        print(f"  Evolution complete.")
        print(f"  Global best fitness (val): {self._global_best_fitness:.4f}")
        print(f"  Global best equation: {self._global_best_eq}")
        return self._global_best_eq.copy()

    # ── GA tournament selection ───────────────────────────────────────────────

    def _tournament_select(self, champions):
        """
        Pick the top two island champions by fitness.
        With GA_TOURNAMENT_K >= 2 this is just ranking.
        Returns (parent_a, parent_b) as Equation objects.
        """
        ranked = sorted(champions, key=lambda x: x[2], reverse=True)
        parent_a = ranked[0][1]
        parent_b = ranked[1][1] if len(ranked) > 1 else ranked[0][1]
        return parent_a, parent_b

    # ── Fitness helper (shared evaluation across all province data) ───────────

    def _eval_fitness(self, equation: Equation, years: List[int]) -> float:
        """Shape-weighted fitness across all province DataFrames."""
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

    # ── Record keeping ────────────────────────────────────────────────────────

    def _record_epoch(self, epoch, champions, migrant, migrant_fitness):
        record = {
            "epoch"             : epoch + 1,
            "global_best_fitness": self._global_best_fitness,
            "migrant_fitness"   : migrant_fitness,
            "island_fitnesses"  : {isl.name: fit for isl, _, fit in champions},
        }
        if migrant is not None:
            record["migrant_equation"] = str(migrant)
            record["migrant_n_terms"]  = migrant.n_terms()
        self.epoch_history.append(record)
