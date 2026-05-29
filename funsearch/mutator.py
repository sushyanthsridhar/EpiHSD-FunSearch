"""
mutator.py
─────────────────────────────────────────────────────────────────────────────
Mutation operators for FunSearch-SINDy (Project 2).

Six operations, selected randomly according to MUTATION_PROBS in config:

  gradient_perturb  — compute dFitness/dξ for each coefficient via finite
                      differences, nudge the coefficient with the steepest
                      positive gradient. Guided local exploitation.

  perturb           — nudge one random coefficient by N(0, σ). Pure random
                      exploration around the current solution.

  add_term          — add a random library term with a small random coef.
                      Grows the equation — structural exploration.

  remove_term       — drop the term with the smallest |coefficient|.
                      Prunes noise — encourages sparsity.

  swap_feature      — replace one term with a different random library term,
                      keeping the coefficient magnitude. Sideways move.

  crossover         — merge terms from two parent equations (p=0.5 each).
                      Recombination across individuals.

Public API
──────────
  mutate(equation, province_dfs, second_parent=None) → Equation
      Selects and applies one mutation operator. Returns a new Equation,
      never modifies the input. second_parent is only used for crossover.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional
import pandas as pd

from config import (
    LIBRARY, MUTATION_PROBS, MAX_TERMS,
    PERTURB_SIGMA, NEW_TERM_SIGMA,
    GRADIENT_STEP, GRADIENT_LR,
)
from equation import Equation
from evaluator import fitness as eval_fitness


# ── Main entry point ──────────────────────────────────────────────────────────

def mutate(
    equation: Equation,
    province_dfs: List[pd.DataFrame],
    second_parent: Optional[Equation] = None,
) -> Equation:
    """
    Select one mutation operator and return a mutated copy of the equation.

    Parameters
    ----------
    equation       : parent equation to mutate
    province_dfs   : province DataFrames for the island (needed for gradient)
    second_parent  : second parent for crossover (if None, crossover falls
                     back to random perturb)

    Returns
    -------
    Equation — a new mutated equation. Input is never modified.
    """
    # Force remove_term if equation has grown too large
    if equation.n_terms() >= MAX_TERMS:
        return _remove_term(equation)

    # Force add_term if equation is empty
    if equation.n_terms() == 0:
        return _add_term(equation)

    # Sample mutation operator according to config probabilities
    ops   = list(MUTATION_PROBS.keys())
    probs = list(MUTATION_PROBS.values())
    # Normalise in case they don't sum to exactly 1
    probs = np.array(probs) / np.sum(probs)
    op    = np.random.choice(ops, p=probs)

    if op == "gradient_perturb":
        return _gradient_perturb(equation, province_dfs)
    elif op == "perturb":
        return _perturb(equation)
    elif op == "add_term":
        return _add_term(equation)
    elif op == "remove_term":
        return _remove_term(equation)
    elif op == "swap_feature":
        return _swap_feature(equation)
    elif op == "crossover":
        if second_parent is not None and second_parent.n_terms() > 0:
            return _crossover(equation, second_parent)
        else:
            return _perturb(equation)  # fallback
    else:
        return _perturb(equation)


# ── Gradient-guided perturbation ──────────────────────────────────────────────

def _gradient_perturb(
    equation: Equation,
    province_dfs: List[pd.DataFrame],
) -> Equation:
    """
    Estimate dFitness/dξⱼ for every coefficient via central finite differences:

        grad(ξⱼ) ≈ [ F(ξⱼ + ε) − F(ξⱼ − ε) ] / 2ε

    Nudge the coefficient with the steepest gradient in the direction of
    improvement (gradient ascent, since we maximise fitness).

    Cost: 2 * n_terms fitness evaluations.
    """
    if equation.n_terms() == 0:
        return _add_term(equation)

    eps = GRADIENT_STEP
    best_grad = 0.0
    best_term = None

    for term, coef in equation.terms.items():
        eq_up   = equation.copy()
        eq_down = equation.copy()
        eq_up.terms[term]   = coef + eps
        eq_down.terms[term] = coef - eps

        f_up   = eval_fitness(eq_up,   province_dfs)
        f_down = eval_fitness(eq_down, province_dfs)

        grad = (f_up - f_down) / (2.0 * eps)

        # Track the term with the steepest absolute gradient
        if abs(grad) > abs(best_grad):
            best_grad = grad
            best_term = term

    new_eq = equation.copy()

    if best_term is None or best_grad == 0.0:
        # Gradient is flat — fall back to random perturb
        return _perturb(equation)

    # Step along gradient + small noise to avoid getting stuck on plateaus
    nudge = GRADIENT_LR * best_grad + np.random.normal(0, PERTURB_SIGMA * 0.3)
    new_eq.terms[best_term] = equation.terms[best_term] + nudge

    # Remove if coefficient collapsed to near zero
    if abs(new_eq.terms[best_term]) < 1e-4:
        del new_eq.terms[best_term]

    return new_eq


# ── Random perturbation ───────────────────────────────────────────────────────

def _perturb(equation: Equation) -> Equation:
    """Nudge one randomly chosen coefficient by N(0, PERTURB_SIGMA)."""
    if equation.n_terms() == 0:
        return _add_term(equation)

    new_eq = equation.copy()
    term   = np.random.choice(list(new_eq.terms.keys()))
    new_eq.terms[term] += np.random.normal(0, PERTURB_SIGMA)

    # Remove if collapsed to near zero
    if abs(new_eq.terms[term]) < 1e-4:
        del new_eq.terms[term]

    return new_eq


# ── Add term ──────────────────────────────────────────────────────────────────

def _add_term(equation: Equation) -> Equation:
    """Add a random library term not already in the equation."""
    available = [t for t in LIBRARY if t not in equation.terms]
    if not available:
        return _perturb(equation)  # library exhausted — perturb instead

    new_eq = equation.copy()
    term   = np.random.choice(available)
    new_eq.terms[term] = np.random.normal(0, NEW_TERM_SIGMA)

    # Resample if too close to zero
    while abs(new_eq.terms[term]) < 1e-4:
        new_eq.terms[term] = np.random.normal(0, NEW_TERM_SIGMA)

    return new_eq


# ── Remove term ───────────────────────────────────────────────────────────────

def _remove_term(equation: Equation) -> Equation:
    """Drop the term with the smallest absolute coefficient."""
    if equation.n_terms() == 0:
        return equation.copy()

    new_eq   = equation.copy()
    weakest  = min(new_eq.terms, key=lambda t: abs(new_eq.terms[t]))
    del new_eq.terms[weakest]

    return new_eq


# ── Swap feature ──────────────────────────────────────────────────────────────

def _swap_feature(equation: Equation) -> Equation:
    """Replace one random term with a different random library term."""
    if equation.n_terms() == 0:
        return _add_term(equation)

    new_eq    = equation.copy()
    old_term  = np.random.choice(list(new_eq.terms.keys()))
    old_coef  = new_eq.terms.pop(old_term)

    available = [t for t in LIBRARY if t not in new_eq.terms]
    if not available:
        # No room — put it back
        new_eq.terms[old_term] = old_coef
        return new_eq

    new_term = np.random.choice(available)
    new_eq.terms[new_term] = old_coef   # keep the coefficient magnitude

    return new_eq


# ── Crossover ─────────────────────────────────────────────────────────────────

def _crossover(parent_a: Equation, parent_b: Equation) -> Equation:
    """
    Merge terms from two parent equations.
    Each term from either parent is included with probability 0.5.
    If a term appears in both, average the coefficients.
    """
    all_terms = set(parent_a.terms) | set(parent_b.terms)
    new_terms = {}

    for term in all_terms:
        in_a = term in parent_a.terms
        in_b = term in parent_b.terms

        if in_a and in_b:
            # Both have it — average coefficients, always include
            new_terms[term] = (parent_a.terms[term] + parent_b.terms[term]) / 2.0
        elif in_a:
            if np.random.random() < 0.5:
                new_terms[term] = parent_a.terms[term]
        else:
            if np.random.random() < 0.5:
                new_terms[term] = parent_b.terms[term]

    # Ensure result is not empty
    if not new_terms:
        return _perturb(parent_a)

    return Equation(new_terms)
