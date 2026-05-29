"""
llm_mutator.py
─────────────────────────────────────────────────────────────────────────────
LLM-backed structural mutation operator for funsearch-ga-sindy.

Two modes controlled by config.LLM_MODE:

  "algorithmic"  (default)
      Uses the same add/remove/swap/crossover operators as the old
      funsearch/mutator.py.  No API calls, fully deterministic given a seed.
      Generates N_LLM_SUGGESTIONS variants by applying different operators.

  "claude"
      Calls the Anthropic Claude API.  The best equation and its fitness
      context are sent as a prompt; Claude proposes a structural
      modification (which term to add or remove).  The suggestion is
      parsed and applied, then coefficients are re-fitted via OLS.
      Requires ANTHROPIC_API_KEY in the environment.

Public API
──────────
  llm_mutate(equation, province_dfs, train_years, n_suggestions)
      → List[Equation]
      Generate n_suggestions structurally diverse variants of equation.
      Coefficients are NOT fitted here; call sindy_fitter.refit() on each.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import re
import json
import numpy as np
from typing import List, Optional

from config import (
    LIBRARY, MUTATION_PROBS, MAX_TERMS,
    PERTURB_SIGMA, NEW_TERM_SIGMA,
    GRADIENT_STEP, GRADIENT_LR,
    LLM_MODE, BASE_FEATURES,
)
from equation import Equation
from evaluator import fitness as eval_fitness


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def llm_mutate(
    equation      : Equation,
    province_dfs  : list,
    train_years   : List[int],
    n_suggestions : int = 3,
) -> List[Equation]:
    """
    Generate structurally diverse variants of the given equation.

    Returns a list of Equation objects whose STRUCTURE has been modified
    but whose coefficients are still placeholder (not OLS-fitted).
    The caller is responsible for calling sindy_fitter.refit() on each.

    Parameters
    ----------
    equation      : best equation selected by GA coordinator
    province_dfs  : list of province DataFrames (for gradient operator)
    train_years   : training years (for gradient operator)
    n_suggestions : how many variants to generate

    Returns
    -------
    List[Equation] of length n_suggestions
    """
    if LLM_MODE == "claude":
        return _claude_mutate(equation, province_dfs, train_years, n_suggestions)
    else:
        return _algorithmic_mutate(equation, province_dfs, n_suggestions)


# ══════════════════════════════════════════════════════════════════════════════
# Mode 1: Algorithmic mutations (default)
# ══════════════════════════════════════════════════════════════════════════════

def _algorithmic_mutate(
    equation     : Equation,
    province_dfs : list,
    n            : int,
) -> List[Equation]:
    """
    Apply n different structural operators to produce n diverse variants.
    Ensures structural diversity by rotating through operator types.
    """
    # Ordered operators — rotate through them for diversity
    operators = [
        _add_term,
        _swap_feature,
        _remove_term,
        lambda eq: _crossover(eq, _add_term(eq)),  # crossover with a grown sibling
        _add_term,  # repeat add for higher weight
    ]
    results = []
    seen_structures = set()

    for i in range(n * 5):   # try up to 5x to get n unique structures
        if len(results) >= n:
            break
        op  = operators[i % len(operators)]
        try:
            candidate = op(equation)
        except Exception:
            candidate = _perturb(equation)

        struct_key = frozenset(candidate.terms.keys())
        if struct_key not in seen_structures:
            seen_structures.add(struct_key)
            results.append(candidate)

    # Pad with perturb copies if we couldn't find enough diverse structures
    while len(results) < n:
        results.append(_perturb(equation))

    return results[:n]


# ── Operator implementations (from original mutator.py) ───────────────────────

def _perturb(equation: Equation) -> Equation:
    if equation.n_terms() == 0:
        return _add_term(equation)
    new_eq = equation.copy()
    term   = np.random.choice(list(new_eq.terms.keys()))
    new_eq.terms[term] += np.random.normal(0, PERTURB_SIGMA)
    if abs(new_eq.terms[term]) < 1e-4:
        del new_eq.terms[term]
    return new_eq


def _add_term(equation: Equation) -> Equation:
    available = [t for t in LIBRARY if t not in equation.terms]
    if not available:
        return _perturb(equation)
    new_eq = equation.copy()
    term   = np.random.choice(available)
    coef   = np.random.normal(0, NEW_TERM_SIGMA)
    while abs(coef) < 1e-4:
        coef = np.random.normal(0, NEW_TERM_SIGMA)
    new_eq.terms[term] = coef
    # Enforce sparsity cap
    if new_eq.n_terms() > MAX_TERMS:
        weakest = min(new_eq.terms, key=lambda t: abs(new_eq.terms[t]))
        del new_eq.terms[weakest]
    return new_eq


def _remove_term(equation: Equation) -> Equation:
    if equation.n_terms() == 0:
        return equation.copy()
    new_eq  = equation.copy()
    weakest = min(new_eq.terms, key=lambda t: abs(new_eq.terms[t]))
    del new_eq.terms[weakest]
    return new_eq


def _swap_feature(equation: Equation) -> Equation:
    if equation.n_terms() == 0:
        return _add_term(equation)
    new_eq   = equation.copy()
    old_term = np.random.choice(list(new_eq.terms.keys()))
    old_coef = new_eq.terms.pop(old_term)
    available = [t for t in LIBRARY if t not in new_eq.terms]
    if not available:
        new_eq.terms[old_term] = old_coef
        return new_eq
    new_eq.terms[np.random.choice(available)] = old_coef
    return new_eq


def _crossover(parent_a: Equation, parent_b: Equation) -> Equation:
    all_terms = set(parent_a.terms) | set(parent_b.terms)
    new_terms = {}
    for term in all_terms:
        in_a = term in parent_a.terms
        in_b = term in parent_b.terms
        if in_a and in_b:
            new_terms[term] = (parent_a.terms[term] + parent_b.terms[term]) / 2.0
        elif in_a and np.random.random() < 0.5:
            new_terms[term] = parent_a.terms[term]
        elif in_b and np.random.random() < 0.5:
            new_terms[term] = parent_b.terms[term]
    if not new_terms:
        return _perturb(parent_a)
    return Equation(new_terms)


# ══════════════════════════════════════════════════════════════════════════════
# Mode 2: Claude API mutations (optional)
# ══════════════════════════════════════════════════════════════════════════════

def _claude_mutate(
    equation      : Equation,
    province_dfs  : list,
    train_years   : List[int],
    n_suggestions : int,
) -> List[Equation]:
    """
    Call the Anthropic Claude API to propose structural modifications to
    the current best equation.

    Claude receives:
      - The equation in human-readable form
      - The full library of available terms
      - Epidemiological context about what each term means
      - Instructions to propose one add or remove operation

    The response is parsed to extract the proposed term action, which is
    applied to produce a candidate equation.  Falls back to algorithmic
    mutation if the API call fails or the response cannot be parsed.
    """
    try:
        import anthropic
    except ImportError:
        print("  [LLM] anthropic package not installed. "
              "Falling back to algorithmic mutation.")
        return _algorithmic_mutate(equation, province_dfs, n_suggestions)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  [LLM] ANTHROPIC_API_KEY not set. "
              "Falling back to algorithmic mutation.")
        return _algorithmic_mutate(equation, province_dfs, n_suggestions)

    client = anthropic.Anthropic(api_key=api_key)

    # Build current term list
    current_terms = "\n".join(
        f"  {coef:+.4f}  {term}"
        for term, coef in sorted(equation.terms.items(),
                                 key=lambda x: -abs(x[1]))
    )
    available_to_add = [t for t in LIBRARY if t not in equation.terms]
    library_excerpt  = ", ".join(available_to_add[:20])
    if len(available_to_add) > 20:
        library_excerpt += f" ... and {len(available_to_add)-20} more"

    prompt = f"""You are helping discover a sparse differential equation for dengue fever incidence
in Dominican Republic provinces.  The target variable is dI/dt (weekly change in
normalised incidence per 100k population).

Current best equation ({equation.n_terms()} terms):
dI/dt =
{current_terms}

Available library terms not yet in the equation:
{library_excerpt}

Key term meanings:
- EIP = exp(0.347*temp - 9.17): mosquito extrinsic incubation period (transmission capacity)
- rolling_incidence_4wk: 4-week rolling mean of incidence (epidemic momentum)
- incidence_lag1, lag2, lag4: past incidence at 1, 2, 4 week delays
- temp_lag4: temperature 4 weeks ago (aligns with mosquito generation time)
- week_sin, week_cos: annual seasonal cycle encoding
- Interaction terms like "EIP rolling_incidence_4wk": product of both features

Propose exactly ONE structural change to improve epidemiological interpretability
or predictive power.  Choose from:
  ADD <term>    : add one term from the available library
  REMOVE <term> : remove one existing term (use exact name from equation)

Respond in this exact JSON format:
{{"action": "ADD" or "REMOVE", "term": "<exact term name>", "reason": "<one sentence>"}}

Do not add any text outside the JSON."""

    results = []
    for attempt in range(n_suggestions * 2):
        if len(results) >= n_suggestions:
            break
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            # Extract JSON
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                continue
            parsed = json.loads(m.group())
            action = parsed.get("action", "").upper()
            term   = parsed.get("term", "").strip()
            reason = parsed.get("reason", "")

            candidate = equation.copy()

            if action == "ADD" and term in LIBRARY and term not in candidate.terms:
                candidate.terms[term] = np.random.normal(0, NEW_TERM_SIGMA)
                if candidate.n_terms() > MAX_TERMS:
                    weakest = min(candidate.terms,
                                  key=lambda t: abs(candidate.terms[t]))
                    del candidate.terms[weakest]
                print(f"  [LLM] ADD {term}  ({reason})")
                results.append(candidate)

            elif action == "REMOVE" and term in candidate.terms:
                del candidate.terms[term]
                print(f"  [LLM] REMOVE {term}  ({reason})")
                results.append(candidate)

        except Exception as e:
            print(f"  [LLM] API call failed: {e}")
            continue

    if len(results) < n_suggestions:
        fallback = _algorithmic_mutate(equation, province_dfs,
                                       n_suggestions - len(results))
        results.extend(fallback)

    return results[:n_suggestions]
