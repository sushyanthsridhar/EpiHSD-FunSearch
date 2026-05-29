"""
equation.py
─────────────────────────────────────────────────────────────────────────────
The Equation class: the fundamental unit of FunSearch-SINDy.

An equation is a sparse linear combination of library terms:

    dI/dt_norm ≈ Σ  coef_i · term_i(X)

Internally stored as dict[str, float].  Terms can be:
  - "bias"                    → constant 1
  - "incidence_lag1"          → df["incidence_lag1"]
  - "incidence_lag1^2"        → df["incidence_lag1"] ** 2
  - "incidence_lag1 EIP"      → df["incidence_lag1"] * df["EIP"]

The string format mirrors the SINDy m1_summary.csv output so that seed
equations can be loaded without loss of information.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Dict

from config import BASE_FEATURES, LIBRARY


class Equation:
    """Sparse linear combination of candidate library terms."""

    def __init__(self, terms: Dict[str, float] | None = None):
        # terms: {term_name: coefficient}  — zero terms are excluded
        self.terms: Dict[str, float] = terms or {}

    # ── Core arithmetic ───────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute predicted dI_dt_norm for every row in df.

        Returns a 1-D numpy array of length len(df).
        Raises ValueError if a term references a column not in df.
        """
        result = np.zeros(len(df), dtype=float)
        for term, coef in self.terms.items():
            result += coef * _term_values(term, df)
        return result

    # ── Parsing ───────────────────────────────────────────────────────────────

    @classmethod
    def parse(cls, s: str) -> "Equation":
        """
        Parse an equation string from m1_summary.csv.

        Format examples (SINDy output style):
            "-1.54 incidence_lag1 +  1.87 incidence_lag2"
            "+0.24 incidence_lag1 EIP"
            "-0.35 EIP incidence_lag4"
            "1.0 bias"
        """
        s = s.strip()
        if not s:
            return cls()

        # ── Normalise ─────────────────────────────────────────────────────────
        # Merge explicit "+-" pairs: "+ -0.35" → "-0.35"
        s = re.sub(r'\+\s*-', '-', s)
        # Collapse all whitespace runs to a single space
        s = re.sub(r'\s+', ' ', s)
        # Merge standalone minus sign with the following coefficient so that
        # "feat - 0.65 feat2" becomes "feat -0.65 feat2".
        # Without this, "-" is collected as part of the feature name and the
        # coefficient sign is lost.  The lookbehind \S ensures we only match
        # a minus that appears as a binary operator (after a word character),
        # not one that is already attached to a number.
        s = re.sub(r'(?<=\S) - (?=\d)', ' -', s)

        terms: Dict[str, float] = {}

        # ── Token-by-token walk ───────────────────────────────────────────────
        # Tokens are space-separated.  A term is: COEF FEAT [FEAT]
        # where COEF is a float (possibly with leading '-' or '+')
        # and FEAT is a word (feature name).
        tokens = s.split(' ')
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if not tok or tok == '+':
                i += 1
                continue

            # Try to parse token as a float coefficient
            try:
                coef = float(tok)
            except ValueError:
                i += 1
                continue

            # Collect subsequent word tokens as feature name(s)
            feature_parts = []
            j = i + 1
            while j < len(tokens):
                t = tokens[j]
                if not t or t == '+':
                    j += 1
                    continue
                # Stop if this token is another number (next coefficient)
                try:
                    float(t)
                    break
                except ValueError:
                    feature_parts.append(t)
                    j += 1

            if feature_parts:
                term = _normalise_term(' '.join(feature_parts))
                if term is not None:
                    terms[term] = terms.get(term, 0.0) + coef

            i = j

        return cls({k: v for k, v in terms.items() if abs(v) > 1e-10})

    # ── Utilities ─────────────────────────────────────────────────────────────

    def copy(self) -> "Equation":
        return Equation(deepcopy(self.terms))

    def n_terms(self) -> int:
        return len(self.terms)

    def sparsity(self) -> float:
        """Fraction of library terms that are zero (higher = sparser)."""
        return 1.0 - len(self.terms) / len(LIBRARY)

    def __repr__(self) -> str:
        if not self.terms:
            return "dI/dt = 0"
        parts = []
        for term, coef in sorted(self.terms.items(), key=lambda x: -abs(x[1])):
            sign = "+" if coef >= 0 else "-"
            parts.append(f"{sign} {abs(coef):.4f} {term}")
        body = "  ".join(parts).lstrip("+ ").strip()
        return f"dI/dt = {body}"

    def to_dict(self) -> Dict[str, float]:
        return dict(self.terms)


# ── Private helpers ───────────────────────────────────────────────────────────

def _term_values(term: str, df: pd.DataFrame) -> np.ndarray:
    """
    Compute the numeric values for a single library term given a DataFrame.

    Handles:
      bias               → 1.0
      single_feature     → df[feature]
      feat^2             → df[feature] ** 2
      feat_a feat_b      → df[feat_a] * df[feat_b]   (SINDy space notation)
      feat_a * feat_b    → same
    """
    term = term.strip()

    # Bias / constant
    if term == "bias":
        return np.ones(len(df), dtype=float)

    # Square term  e.g. "EIP^2"
    if term.endswith("^2"):
        feat = term[:-2]
        if feat not in df.columns:
            raise ValueError(f"Feature '{feat}' not found in DataFrame.")
        return df[feat].values.astype(float) ** 2

    # Interaction term with '*' separator  e.g. "EIP*incidence_lag1"
    if "*" in term:
        parts = [p.strip() for p in term.split("*")]
        result = np.ones(len(df), dtype=float)
        for p in parts:
            if p not in df.columns:
                raise ValueError(f"Feature '{p}' not found in DataFrame.")
            result *= df[p].values.astype(float)
        return result

    # Interaction term with space separator  e.g. "EIP incidence_lag1"
    parts = term.split()
    if len(parts) == 2:
        a, b = parts
        if a not in df.columns:
            raise ValueError(f"Feature '{a}' not found in DataFrame.")
        if b not in df.columns:
            raise ValueError(f"Feature '{b}' not found in DataFrame.")
        return df[a].values.astype(float) * df[b].values.astype(float)

    # Single feature
    if len(parts) == 1:
        if term not in df.columns:
            raise ValueError(f"Feature '{term}' not found in DataFrame.")
        return df[term].values.astype(float)

    raise ValueError(f"Cannot parse term: '{term}'")


def _normalise_term(raw: str) -> str | None:
    """
    Convert a raw term string (from SINDy CSV or any format) to the
    canonical library key used in config.LIBRARY.

    Returns None if the term cannot be matched to the library.
    """
    raw = raw.strip()

    if raw == "bias" or raw == "1":
        return "bias"

    # Square with ^ notation: "EIP^2" → already canonical
    if raw.endswith("^2"):
        feat = raw[:-2]
        if feat in BASE_FEATURES:
            return raw
        return None

    # Square with repetition: "EIP EIP" → "EIP^2"
    parts = raw.split()
    if len(parts) == 2 and parts[0] == parts[1] and parts[0] in BASE_FEATURES:
        return f"{parts[0]}^2"

    # Interaction — normalise order to match LIBRARY (combinations are sorted)
    if len(parts) == 2:
        a, b = parts
        if a in BASE_FEATURES and b in BASE_FEATURES:
            # Return in same order as config.CROSS_TERMS (combinations preserves insertion order)
            idx_a = BASE_FEATURES.index(a)
            idx_b = BASE_FEATURES.index(b)
            if idx_a < idx_b:
                return f"{a} {b}"
            else:
                return f"{b} {a}"
        return None

    # Single base feature
    if len(parts) == 1 and raw in BASE_FEATURES:
        return raw

    return None
