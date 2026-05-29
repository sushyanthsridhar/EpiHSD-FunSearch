"""
run_funsearch.py
────────────────────────────────────────────────────────────
FunSearch for EpiHSD: evolutionary discovery of optimal feature
libraries for the EpiHSD dengue forecasting framework.

ALGORITHM
─────────
Island model with a programmatic mutator standing in for the LLM.
Each island maintains a ranked population of Programs. In every
iteration each island:
  1. Samples a parent (tournament selection)
  2. Mutates it into a child (or occasionally crosses two parents)
  3. Evaluates the child on the inner-val fitness function
  4. Adds the child to the island population, evicting the worst

Every MIGRATE_EVERY iterations the best program on each island is
shared to all other islands, injecting diversity.

MUTATOR DESIGN
──────────────
The mutator is the core of the search. Rather than uniform random
ops, it uses a weighted operator menu:

  perturb_term   nudge one parameter of an existing term    w=3
  replace_term   swap one term for a fresh random one       w=2
  add_term       append a new random term                   w=2
  retype_term    change term_type, preserve compatible params w=1
  n_lags_up/down adjust lag depth by 1                      w=1 each
  remove_term    drop one term (weight scales with fill)
  n_lags_jump    change lag depth by +/-2                   w=0.5

With 15% probability two ops are applied to the same child.

ISLAND DIVERSITY
────────────────
Each island is seeded from a differently biased variant of the
base seed so they start in different corners of the search space:
  Island 0   conservative   low n_lags, no extra terms
  Island 1   default        seed n_lags=5, no extra terms
  Island 2   deep-lag       high n_lags, no extra terms
  Island 3   nonlinear      mid n_lags + 3 random extra terms

Set N_ISLANDS=3 to drop island 3.

PROGRAM REPRESENTATION
──────────────────────
A Program specifies:
  n_lags      : lag depth for the autoregressive state vector
  extra_terms : list of targeted nonlinear features appended to
                the linear lag matrix before the STLSQ fit

TERM TYPES
──────────
  inc_pow      incidence_lag_k ^ power   (p in 2,3)
  inc_z_prod   incidence_lag_ki * z_j_lag_kz
  cross_inc    incidence_lag_k1 * incidence_lag_k2
  z_sq         z_j_lag_k ^ 2
  z_prod       z_i_lag_k * z_j_lag_k  (cross-latent coupling)
  inc_diff     incidence_lag_k1 - incidence_lag_k2  (epidemic momentum)
  z_cube       z_j_lag_k ^ 3  (stronger nonlinearity)
  seasonal     sin or cos of week/52 * 2pi  (explicit annual cycle)

FITNESS FUNCTION
────────────────
Inner split: fit on 2015-2017, evaluate on 2018-2019.
Val (2020-2021) and test (2022-2023) are completely sealed.
Fitness = province-mean incidence R squared on 2018-2019.

OUTPUTS (weekly-pipeline/)
──────────────────────────
  funsearch_history.csv    per-iteration scores and program specs
  funsearch_best.json      best program found across all islands
  funsearch_islands.json   final island populations
────────────────────────────────────────────────────────────
"""

import os, json, copy, random, uuid, time, hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from dataclasses import dataclass, field
from typing import List, Dict, Any

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
DATA_CSV = os.path.join(ROOT, "extended_input_normalized.csv")

LATENT_DIM  = 6
LATENT_CSV  = os.path.join(HERE, "latents", f"latent_weekly_dim{LATENT_DIM}.csv")
MODEL_PATH  = os.path.join(HERE, "models", f"best_model_weekly_dim{LATENT_DIM}.pt")
CLUSTER_CSV = os.path.join(HERE, "province_clusters.csv")

# ── FunSearch island config ────────────────────────────────
# Set N_ISLANDS to 3 or 4. 4 gives more diversity at modest extra cost.
# Each island is seeded with a different bias so they explore different
# regions of the search space from the start.
N_ISLANDS     = 4     # number of evolutionary islands (try 3 or 4)
ISLAND_SIZE   = 5     # programs kept per island (best N survive)
N_ITERATIONS  = 80    # outer-loop iterations
MIGRATE_EVERY = 10    # share best program across islands every N iters
MAX_EXTRA     = 8     # cap on extra nonlinear terms per program
RANDOM_SEED   = 42

# ── EpiHSD settings fixed across all evaluations ──────────
POLY_DEG            = 1
THRESH_GLOBAL       = 0.05
THRESH_CLUSTER      = 0.10
THRESH_PROVINCE     = 0.08
THRESH_INC_GLOBAL   = 0.04
THRESH_INC_CLUSTER  = 0.10
THRESH_INC_PROVINCE = 100.0   # no province incidence delta
LOG_INCIDENCE       = True
STLSQ_ITERS         = 30

# ── Data split (test years NEVER seen during search) ───────
INNER_TRAIN = [2015, 2016, 2017]   # EpiHSD fit inside FunSearch
INNER_VAL   = [2018, 2019]         # fitness evaluated here
TRAIN_YEARS = list(range(2015, 2020))
VAL_YEARS   = [2020, 2021]
TEST_YEARS  = [2022, 2023]         # SEALED until final evaluation

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ════════════════════════════════════════════════════════════
# DATA LOADING (done once at module level)
# ════════════════════════════════════════════════════════════
print("FunSearch  loading data and encoding latents...")

ACTS = {"relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "elu": nn.ELU}

class WeeklyAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims,
                 dropout, activation, use_batchnorm):
        super().__init__()
        act = ACTS[activation]
        enc_dims = [input_dim] + hidden_dims + [latent_dim]
        enc_layers = []
        for i in range(len(enc_dims) - 1):
            enc_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i < len(enc_dims) - 2:
                if use_batchnorm:
                    enc_layers.append(nn.BatchNorm1d(enc_dims[i + 1]))
                enc_layers.append(act())
                if dropout > 0.0:
                    enc_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*enc_layers)
        dec_dims = list(reversed(enc_dims))
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                if use_batchnorm:
                    dec_layers.append(nn.BatchNorm1d(dec_dims[i + 1]))
                dec_layers.append(act())
                if dropout > 0.0:
                    dec_layers.append(nn.Dropout(dropout))
            else:
                dec_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec_layers)
    def encode(self, x): return self.encoder(x)

raw = pd.read_csv(DATA_CSV)
raw = raw.sort_values(["province", "year", "week"]).reset_index(drop=True)
ID_COLS      = ["province", "year", "month", "week"]
FEATURE_COLS = [c for c in raw.columns if c not in ID_COLS]
raw["time_sin"] = np.sin(2.0 * np.pi * raw["week"] / 52.0)
raw["time_cos"] = np.cos(2.0 * np.pi * raw["week"] / 52.0)
INPUT_FEATURES = FEATURE_COLS + ["time_sin", "time_cos"]
INPUT_DIM      = len(INPUT_FEATURES)
train_mask     = raw["year"].isin(TRAIN_YEARS)
col_means      = raw.loc[train_mask, INPUT_FEATURES].mean()
raw[INPUT_FEATURES] = raw[INPUT_FEATURES].fillna(col_means)

ae = WeeklyAE(INPUT_DIM, LATENT_DIM, [512, 256], 0.158, "leakyrelu", True)
ae.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
ae.eval()

# Encode all non-training years (val + test) and merge with saved latents
z_cols = [f"z{i+1}" for i in range(LATENT_DIM)]
extra_mask = ~raw["year"].isin(TRAIN_YEARS)
X_extra    = torch.tensor(raw.loc[extra_mask, INPUT_FEATURES].values.astype(np.float32))
with torch.no_grad():
    Z_extra = ae.encode(X_extra).numpy()
extra_lat              = pd.DataFrame(Z_extra, columns=z_cols)
extra_lat["province"]  = raw.loc[extra_mask, "province"].values
extra_lat["year"]      = raw.loc[extra_mask, "year"].values
extra_lat["week"]      = raw.loc[extra_mask, "week"].values

saved  = pd.read_csv(LATENT_CSV)
latent = pd.concat([saved, extra_lat], ignore_index=True)
latent = latent.merge(
    raw[["province", "year", "week", "incidence"]],
    on=["province", "year", "week"], how="left")
latent = latent.sort_values(["province", "year", "week"]).reset_index(drop=True)

if LOG_INCIDENCE:
    latent["incidence"] = np.log1p(latent["incidence"].clip(lower=0))

clusters = pd.read_csv(CLUSTER_CSV)[["province", "cluster_id"]]
latent   = latent.merge(clusters, on="province", how="left")

STATE_COLS = z_cols + ["incidence"]
STATE_DIM  = len(STATE_COLS)
INC_COL    = STATE_DIM - 1
CLUSTER_IDS = sorted(latent["cluster_id"].dropna().unique().astype(int))

print(f"  Loaded {len(latent)} latent rows  "
      f"provinces={latent['province'].nunique()}  "
      f"clusters={len(CLUSTER_IDS)}\n")

# ════════════════════════════════════════════════════════════
# PROGRAM REPRESENTATION
# ════════════════════════════════════════════════════════════

@dataclass
class ExtraTerm:
    """One targeted nonlinear feature appended to the lag matrix."""
    term_type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def compute(self, X_lag: np.ndarray, n_lags: int,
                weeks: "np.ndarray | None" = None) -> "np.ndarray | None":
        """Return (n_rows, 1) float32 array, or None on error.
        weeks: optional 1-D array of week numbers aligned to X_lag rows,
               required only by the seasonal term type.
        """
        try:
            tt = self.term_type
            p  = self.params
            ml = n_lags
            if tt == "inc_pow":
                k  = min(p.get("lag", 0), ml)
                pw = p.get("power", 2)
                v  = X_lag[:, STATE_DIM * k + INC_COL] ** pw
            elif tt == "inc_z_prod":
                ki = min(p.get("lag_inc", 0), ml)
                kz = min(p.get("lag_z",   0), ml)
                j  = min(p.get("z_idx",   0), LATENT_DIM - 1)
                v  = X_lag[:, STATE_DIM * ki + INC_COL] * X_lag[:, STATE_DIM * kz + j]
            elif tt == "cross_inc":
                k1 = min(p.get("lag1", 0), ml)
                k2 = min(p.get("lag2", 1), ml)
                v  = X_lag[:, STATE_DIM * k1 + INC_COL] * X_lag[:, STATE_DIM * k2 + INC_COL]
            elif tt == "z_sq":
                k = min(p.get("lag", 0), ml)
                j = min(p.get("z_idx", 0), LATENT_DIM - 1)
                v = X_lag[:, STATE_DIM * k + j] ** 2
            elif tt == "z_prod":
                # cross-latent product: z_i(t-k) * z_j(t-k)
                k  = min(p.get("lag", 0), ml)
                i  = min(p.get("z_idx1", 0), LATENT_DIM - 1)
                j  = min(p.get("z_idx2", 1), LATENT_DIM - 1)
                v  = X_lag[:, STATE_DIM * k + i] * X_lag[:, STATE_DIM * k + j]
            elif tt == "inc_diff":
                # epidemic momentum: I(t-k1) - I(t-k2), proxy for Rt change
                k1 = min(p.get("lag1", 0), ml)
                k2 = min(p.get("lag2", 1), ml)
                v  = (X_lag[:, STATE_DIM * k1 + INC_COL]
                      - X_lag[:, STATE_DIM * k2 + INC_COL])
            elif tt == "z_cube":
                # cubic latent: stronger nonlinearity than z_sq
                k = min(p.get("lag", 0), ml)
                j = min(p.get("z_idx", 0), LATENT_DIM - 1)
                v = X_lag[:, STATE_DIM * k + j] ** 3
            elif tt == "seasonal":
                # explicit annual cycle via sin or cos of week index
                if weeks is None:
                    return None
                use_cos = p.get("use_cos", 0)
                angle   = 2.0 * np.pi * weeks / 52.0
                v = np.cos(angle) if use_cos else np.sin(angle)
            else:
                return None
            return v.astype(np.float32).reshape(-1, 1)
        except Exception:
            return None

    def to_dict(self):
        return {"term_type": self.term_type, "params": dict(self.params)}

    @staticmethod
    def from_dict(d):
        return ExtraTerm(d["term_type"], d["params"])


@dataclass
class Program:
    """An EpiHSD feature construction program."""
    program_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    n_lags: int = 5
    extra_terms: List[ExtraTerm] = field(default_factory=list)
    score: float = float("-inf")
    generation: int = 0
    parent_id: str = "seed"

    def fingerprint(self) -> str:
        key = json.dumps(self.to_spec(), sort_keys=True)
        return hashlib.md5(key.encode()).hexdigest()

    def to_spec(self) -> dict:
        return {"n_lags": self.n_lags,
                "extra_terms": [t.to_dict() for t in self.extra_terms]}

    def to_dict(self) -> dict:
        return {"program_id": self.program_id, "n_lags": self.n_lags,
                "extra_terms": [t.to_dict() for t in self.extra_terms],
                "score": self.score, "generation": self.generation,
                "parent_id": self.parent_id}

    @staticmethod
    def from_dict(d: dict) -> "Program":
        return Program(
            program_id  = d["program_id"],
            n_lags      = d["n_lags"],
            extra_terms = [ExtraTerm.from_dict(t) for t in d["extra_terms"]],
            score       = d["score"],
            generation  = d.get("generation", 0),
            parent_id   = d.get("parent_id", "?"))

    def describe(self) -> str:
        extras = [f"{t.term_type}({t.params})" for t in self.extra_terms]
        return (f"n_lags={self.n_lags}  "
                f"extras=[{', '.join(extras) if extras else 'none'}]  "
                f"score={self.score:+.4f}")


# ════════════════════════════════════════════════════════════
# MUTATOR
# ════════════════════════════════════════════════════════════

TERM_TYPES = ["inc_pow", "inc_z_prod", "cross_inc", "z_sq",
              "z_prod", "inc_diff", "z_cube", "seasonal"]

def _sample_term(n_lags: int) -> ExtraTerm:
    """Sample a completely random new term."""
    tt = random.choice(TERM_TYPES)
    if tt == "inc_pow":
        return ExtraTerm("inc_pow", {
            "lag":   random.randint(0, n_lags),
            "power": random.choice([2, 3])})
    elif tt == "inc_z_prod":
        return ExtraTerm("inc_z_prod", {
            "lag_inc": random.randint(0, n_lags),
            "lag_z":   random.randint(0, n_lags),
            "z_idx":   random.randint(0, LATENT_DIM - 1)})
    elif tt == "cross_inc":
        k1 = random.randint(0, n_lags)
        k2 = random.randint(0, n_lags)
        return ExtraTerm("cross_inc", {"lag1": k1, "lag2": k2})
    elif tt == "z_sq":
        return ExtraTerm("z_sq", {
            "lag":   random.randint(0, n_lags),
            "z_idx": random.randint(0, LATENT_DIM - 1)})
    elif tt == "z_prod":
        i = random.randint(0, LATENT_DIM - 1)
        j = (i + random.randint(1, LATENT_DIM - 1)) % LATENT_DIM
        return ExtraTerm("z_prod", {
            "lag":    random.randint(0, n_lags),
            "z_idx1": i, "z_idx2": j})
    elif tt == "inc_diff":
        return ExtraTerm("inc_diff", {
            "lag1": random.randint(0, n_lags),
            "lag2": random.randint(0, n_lags)})
    elif tt == "z_cube":
        return ExtraTerm("z_cube", {
            "lag":   random.randint(0, n_lags),
            "z_idx": random.randint(0, LATENT_DIM - 1)})
    else:  # seasonal
        return ExtraTerm("seasonal", {"use_cos": random.randint(0, 1)})


def _perturb_term(term: ExtraTerm, n_lags: int) -> ExtraTerm:
    """
    Fine-grained mutation: pick one parameter of an existing term and
    nudge it by +/-1 (or flip a binary). This explores the neighbourhood
    of a term rather than replacing it wholesale.
    """
    t  = copy.deepcopy(term)
    p  = t.params
    tt = t.term_type
    d  = random.choice([-1, 1])

    if tt == "inc_pow":
        if random.random() < 0.5:
            p["lag"] = max(0, min(p["lag"] + d, n_lags))
        else:
            p["power"] = 3 if p["power"] == 2 else 2

    elif tt == "inc_z_prod":
        key = random.choice(["lag_inc", "lag_z", "z_idx"])
        if key == "z_idx":
            p["z_idx"] = (p["z_idx"] + d) % LATENT_DIM
        else:
            p[key] = max(0, min(p[key] + d, n_lags))

    elif tt == "cross_inc":
        key = random.choice(["lag1", "lag2"])
        p[key] = max(0, min(p[key] + d, n_lags))

    elif tt == "z_sq":
        if random.random() < 0.5:
            p["lag"] = max(0, min(p["lag"] + d, n_lags))
        else:
            p["z_idx"] = (p["z_idx"] + d) % LATENT_DIM

    elif tt == "z_prod":
        key = random.choice(["lag", "z_idx1", "z_idx2"])
        if key == "lag":
            p["lag"] = max(0, min(p["lag"] + d, n_lags))
        else:
            p[key] = (p[key] + d) % LATENT_DIM

    elif tt == "inc_diff":
        key = random.choice(["lag1", "lag2"])
        p[key] = max(0, min(p[key] + d, n_lags))

    elif tt == "z_cube":
        if random.random() < 0.5:
            p["lag"] = max(0, min(p["lag"] + d, n_lags))
        else:
            p["z_idx"] = (p["z_idx"] + d) % LATENT_DIM

    elif tt == "seasonal":
        p["use_cos"] = 1 - p.get("use_cos", 0)  # flip sin <-> cos

    return t


def _retype_term(term: ExtraTerm, n_lags: int) -> ExtraTerm:
    """
    Change the term_type while preserving compatible parameters where
    possible. Extracts the most reusable values from the old params and
    maps them onto the new type. Falls back to fresh samples where no
    natural mapping exists.
    """
    old = term.term_type
    candidates = [t for t in TERM_TYPES if t != old]
    new_tt = random.choice(candidates)
    p   = term.params
    # pull the most reusable values from old params
    lag = max(0, min(
        p.get("lag", p.get("lag_inc", p.get("lag1", random.randint(0, n_lags)))),
        n_lags))
    z   = p.get("z_idx", p.get("z_idx1", random.randint(0, LATENT_DIM - 1)))

    if new_tt == "inc_pow":
        return ExtraTerm("inc_pow", {
            "lag": lag, "power": random.choice([2, 3])})
    elif new_tt == "inc_z_prod":
        return ExtraTerm("inc_z_prod", {
            "lag_inc": lag,
            "lag_z":   max(0, min(p.get("lag_z", p.get("lag2",
                           random.randint(0, n_lags))), n_lags)),
            "z_idx":   z})
    elif new_tt == "cross_inc":
        return ExtraTerm("cross_inc", {
            "lag1": lag,
            "lag2": max(0, min(p.get("lag_z", p.get("lag2",
                       random.randint(0, n_lags))), n_lags))})
    elif new_tt == "z_sq":
        return ExtraTerm("z_sq", {"lag": lag, "z_idx": z})
    elif new_tt == "z_prod":
        j = (z + random.randint(1, LATENT_DIM - 1)) % LATENT_DIM
        return ExtraTerm("z_prod", {"lag": lag, "z_idx1": z, "z_idx2": j})
    elif new_tt == "inc_diff":
        lag2 = max(0, min(p.get("lag2", random.randint(0, n_lags)), n_lags))
        return ExtraTerm("inc_diff", {"lag1": lag, "lag2": lag2})
    elif new_tt == "z_cube":
        return ExtraTerm("z_cube", {"lag": lag, "z_idx": z})
    else:  # seasonal
        return ExtraTerm("seasonal", {"use_cos": random.randint(0, 1)})


def mutate(parent: Program, generation: int) -> Program:
    """
    Return a mutated child of parent.

    Operator menu and weights:
      perturb_term   nudge one parameter of an existing term       weight 3
      replace_term   swap one term for a freshly sampled one       weight 2
      retype_term    change term type, preserve compatible params   weight 1
      add_term       append a new random term                       weight 2
      remove_term    drop one term                                  weight scales with fill
      n_lags_up      increment n_lags by 1                         weight 1
      n_lags_down    decrement n_lags by 1                         weight 1
      n_lags_jump    change n_lags by +/-2                         weight 0.5

    With 15% probability a second independent op is applied to the same
    child, giving double mutations that are especially useful early on.
    """
    child = copy.deepcopy(parent)
    child.program_id = str(uuid.uuid4())[:8]
    child.score      = float("-inf")
    child.generation = generation
    child.parent_id  = parent.program_id

    # Number of ops to apply (occasionally double-mutate)
    n_ops = 2 if random.random() < 0.15 else 1

    for _ in range(n_ops):
        n = len(child.extra_terms)

        ops: List[str] = ["n_lags_up", "n_lags_down", "add_term", "n_lags_jump"]
        wts: List[float] = [1.0, 1.0, 2.0, 0.5]

        if n > 0:
            ops += ["perturb_term", "replace_term", "retype_term"]
            wts  += [3.0, 2.0, 1.0]
            # remove_term becomes more attractive as the library fills up
            ops.append("remove_term")
            wts.append(max(0.5, 2.0 * n / MAX_EXTRA))

        total = sum(wts)
        wts   = [w / total for w in wts]
        op    = random.choices(ops, weights=wts, k=1)[0]

        if op == "n_lags_up":
            child.n_lags = min(child.n_lags + 1, 8)

        elif op == "n_lags_down":
            child.n_lags = max(child.n_lags - 1, 2)

        elif op == "n_lags_jump":
            delta = random.choice([-2, 2])
            child.n_lags = max(2, min(child.n_lags + delta, 8))

        elif op == "add_term":
            if len(child.extra_terms) < MAX_EXTRA:
                child.extra_terms.append(_sample_term(child.n_lags))

        elif op == "remove_term":
            if child.extra_terms:
                child.extra_terms.pop(random.randrange(len(child.extra_terms)))

        elif op == "replace_term":
            if child.extra_terms:
                idx = random.randrange(len(child.extra_terms))
                child.extra_terms[idx] = _sample_term(child.n_lags)

        elif op == "perturb_term":
            if child.extra_terms:
                idx = random.randrange(len(child.extra_terms))
                child.extra_terms[idx] = _perturb_term(
                    child.extra_terms[idx], child.n_lags)

        elif op == "retype_term":
            if child.extra_terms:
                idx = random.randrange(len(child.extra_terms))
                child.extra_terms[idx] = _retype_term(
                    child.extra_terms[idx], child.n_lags)

    return child


def crossover(p1: Program, p2: Program, generation: int) -> Program:
    """Mix extra_terms from two parents, keep n_lags from p1."""
    child = copy.deepcopy(p1)
    child.program_id = str(uuid.uuid4())[:8]
    child.score      = float("-inf")
    child.generation = generation
    child.parent_id  = f"{p1.program_id}+{p2.program_id}"
    combined = list(p1.extra_terms)
    for t in p2.extra_terms:
        if random.random() < 0.5 and len(combined) < MAX_EXTRA:
            combined.append(copy.deepcopy(t))
    child.extra_terms = combined
    return child


# ════════════════════════════════════════════════════════════
# FEATURE BUILDER USING PROGRAM
# ════════════════════════════════════════════════════════════

def build_program_features(Z_sc: np.ndarray,
                            program: Program,
                            weeks: "np.ndarray | None" = None):
    """Build (Theta, target) for one province sequence using program spec.

    weeks: optional 1-D array of week-of-year values for every row of Z_sc,
           used by seasonal terms. Pass grp['week'].values from the caller.
    """
    T, K   = Z_sc.shape
    n_lags = program.n_lags
    if T <= n_lags + 1:
        return None, None
    lag_vecs = []
    for lag in range(n_lags + 1):
        lag_vecs.append(Z_sc[n_lags - lag: T - 1 - lag])
    X_lag = np.concatenate(lag_vecs, axis=1).astype(np.float32)
    # Row i of X_lag corresponds to time index (n_lags + i) in Z_sc.
    # Pass the matching slice of weeks so seasonal terms are correctly aligned.
    weeks_lag = (weeks[n_lags: T - 1].astype(np.float32)
                 if weeks is not None else None)
    extra_cols = []
    for term in program.extra_terms:
        col = term.compute(X_lag, n_lags, weeks=weeks_lag)
        if col is not None and col.shape[0] == X_lag.shape[0]:
            extra_cols.append(col)
    if extra_cols:
        X_lag = np.concatenate([X_lag] + extra_cols, axis=1)
    poly_loc = PolynomialFeatures(degree=POLY_DEG, include_bias=True)
    Theta    = poly_loc.fit_transform(X_lag).astype(np.float32)
    target   = Z_sc[n_lags + 1:].astype(np.float32)
    return Theta, target


# ════════════════════════════════════════════════════════════
# STLSQ
# ════════════════════════════════════════════════════════════

def stlsq(Theta: np.ndarray, Y: np.ndarray,
           threshold, n_iters: int) -> np.ndarray:
    W = np.linalg.lstsq(Theta, Y, rcond=None)[0]
    tv = np.broadcast_to(np.atleast_1d(threshold), (Y.shape[1],))
    for _ in range(n_iters):
        for k in range(Y.shape[1]):
            small    = np.abs(W[:, k]) < tv[k]
            W[small, k] = 0.0
            active   = ~small
            if active.sum() == 0:
                continue
            W[active, k] = np.linalg.lstsq(
                Theta[:, active], Y[:, k], rcond=None)[0]
    return W


# ════════════════════════════════════════════════════════════
# EVALUATOR
# ════════════════════════════════════════════════════════════

EVAL_CACHE: Dict[str, float] = {}

def evaluate(program: Program, verbose: bool = False) -> float:
    """
    Run EpiHSD on INNER_TRAIN and measure province-mean incidence R2
    on INNER_VAL. Returns the fitness score. Caches by fingerprint.
    Test years 2022-2023 are never touched here.
    """
    fp = program.fingerprint()
    if fp in EVAL_CACHE:
        return EVAL_CACHE[fp]

    n_lags = program.n_lags

    # Fit scaler on inner training data only
    inner_train_rows = latent[latent["year"].isin(INNER_TRAIN)][STATE_COLS].values
    scaler = StandardScaler()
    scaler.fit(inner_train_rows)

    thresh_g = np.array([THRESH_GLOBAL]   * LATENT_DIM + [THRESH_INC_GLOBAL])
    thresh_c = np.array([THRESH_CLUSTER]  * LATENT_DIM + [THRESH_INC_CLUSTER])
    thresh_p = np.array([THRESH_PROVINCE] * LATENT_DIM + [THRESH_INC_PROVINCE])

    # ── Level 1: global equation ─────────────────────────
    th_all, tgt_all = [], []
    for prov, grp in latent[latent["year"].isin(INNER_TRAIN)].groupby("province"):
        grp = grp.sort_values(["year", "week"]).reset_index(drop=True)
        Z_sc = scaler.transform(grp[STATE_COLS].values.astype(np.float32))
        Th, tgt = build_program_features(Z_sc, program, weeks=grp["week"].values)
        if Th is not None:
            th_all.append(Th); tgt_all.append(tgt)
    if not th_all:
        EVAL_CACHE[fp] = float("-inf")
        return float("-inf")
    Theta_tr = np.vstack(th_all)
    Y_tr     = np.vstack(tgt_all)
    n_terms  = Theta_tr.shape[1]
    try:
        W_global = stlsq(Theta_tr, Y_tr, thresh_g, STLSQ_ITERS)
    except Exception:
        EVAL_CACHE[fp] = float("-inf")
        return float("-inf")

    # ── Level 2: cluster equations ───────────────────────
    W_clusters: Dict[int, np.ndarray] = {}
    for cid in CLUSTER_IDS:
        c_df = latent[(latent["cluster_id"] == cid) &
                      (latent["year"].isin(INNER_TRAIN))]
        th_c, tgt_c = [], []
        for prov, grp in c_df.groupby("province"):
            grp = grp.sort_values(["year", "week"]).reset_index(drop=True)
            Z_sc = scaler.transform(grp[STATE_COLS].values.astype(np.float32))
            Th, tgt = build_program_features(Z_sc, program, weeks=grp["week"].values)
            if Th is not None:
                th_c.append(Th); tgt_c.append(tgt)
        if not th_c:
            W_clusters[cid] = W_global.copy()
            continue
        Tc = np.vstack(th_c); Yc = np.vstack(tgt_c)
        Rc = Yc - Tc @ W_global
        try:
            dWc = stlsq(Tc, Rc, thresh_c, STLSQ_ITERS)
        except Exception:
            dWc = np.zeros_like(W_global)
        W_clusters[cid] = W_global + dWc

    # ── Level 3: province equations ──────────────────────
    W_provinces: Dict[str, np.ndarray] = {}
    for prov, grp in latent.groupby("province"):
        cid = int(grp["cluster_id"].iloc[0])
        W_c = W_clusters.get(cid, W_global)
        grp_tr = (grp[grp["year"].isin(INNER_TRAIN)]
                  .sort_values(["year", "week"]).reset_index(drop=True))
        if len(grp_tr) <= n_lags + 1:
            W_provinces[prov] = W_c.copy()
            continue
        Z_sc = scaler.transform(grp_tr[STATE_COLS].values.astype(np.float32))
        Th_p, Y_p = build_program_features(Z_sc, program,
                                            weeks=grp_tr["week"].values)
        if Th_p is None or len(Th_p) < n_terms:
            W_provinces[prov] = W_c.copy()
        else:
            Rp = Y_p - Th_p @ W_c
            try:
                dWp = stlsq(Th_p, Rp, thresh_p, STLSQ_ITERS)
            except Exception:
                dWp = np.zeros_like(W_c)
            W_provinces[prov] = W_c + dWp

    # ── Inner-val fitness: province-mean incidence R2 ────
    r2_list = []
    for prov, grp in latent[latent["year"].isin(INNER_VAL)].groupby("province"):
        grp_s = grp.sort_values(["year", "week"]).reset_index(drop=True)
        if len(grp_s) <= n_lags:
            continue
        Z_s    = grp_s[STATE_COLS].values.astype(np.float32)
        Z_s_sc = scaler.transform(Z_s)
        Th_s, Y_s = build_program_features(Z_s_sc, program,
                                            weeks=grp_s["week"].values)
        if Th_s is None:
            continue
        cid = int(latent.loc[latent["province"] == prov, "cluster_id"].iloc[0])
        W_p = W_provinces.get(prov, W_clusters.get(cid, W_global))
        Y_hat_sc = Th_s @ W_p
        Y_hat_raw = scaler.inverse_transform(Y_hat_sc)
        Y_s_raw   = scaler.inverse_transform(Y_s)
        if LOG_INCIDENCE:
            Y_hat_raw[:, INC_COL] = np.expm1(Y_hat_raw[:, INC_COL])
            Y_s_raw[:,   INC_COL] = np.expm1(Y_s_raw[:,   INC_COL])
        r2 = float(r2_score(Y_s_raw[:, INC_COL], Y_hat_raw[:, INC_COL]))
        r2_list.append(r2)

    score = float(np.mean(r2_list)) if r2_list else float("-inf")
    EVAL_CACHE[fp] = score
    return score


# ════════════════════════════════════════════════════════════
# ISLAND
# ════════════════════════════════════════════════════════════

class Island:
    """
    One evolutionary island.

    Each island has a bias_seed that shapes its initial programs so that
    islands start from different corners of the search space:

      Island 0  conservative   low n_lags (3-4),  no extra terms
      Island 1  default        n_lags=5,           seed program
      Island 2  deep-lag       high n_lags (6-8),  no extra terms
      Island 3  nonlinear      n_lags=4-5,         2-3 extra terms pre-seeded

    With only 3 islands use bias_seeds 0, 1, 2. With 4 add bias_seed 3.
    """

    # Per-island bias config: (n_lags_range, n_init_extra_terms)
    BIAS_TABLE = {
        0: (range(3, 5),  0),   # conservative
        1: (range(5, 6),  0),   # default, no extra terms
        2: (range(6, 9),  0),   # deep-lag exploration
        3: (range(4, 6),  3),   # nonlinear heavy
    }

    def __init__(self, island_id: int, capacity: int = ISLAND_SIZE):
        self.island_id  = island_id
        self.capacity   = capacity
        self.programs: List[Program] = []
        # bias determines the first seed variant for this island
        bias_key = island_id % len(self.BIAS_TABLE)
        lags_rng, n_init_extra = self.BIAS_TABLE[bias_key]
        self._init_n_lags   = random.choice(list(lags_rng))
        self._init_n_extra  = n_init_extra

    def make_biased_seed(self, base_seed: Program, generation: int) -> Program:
        """
        Return a variant of base_seed shaped by this island's bias.
        Does NOT evaluate; caller must set score.
        """
        child = copy.deepcopy(base_seed)
        child.program_id = str(uuid.uuid4())[:8]
        child.score      = float("-inf")
        child.generation = generation
        child.parent_id  = base_seed.program_id
        child.n_lags     = self._init_n_lags
        # pre-load extra terms for the nonlinear island
        child.extra_terms = []
        for _ in range(self._init_n_extra):
            child.extra_terms.append(_sample_term(child.n_lags))
        return child

    def add(self, program: Program):
        """Insert program, keep only the top capacity by score."""
        self.programs.append(program)
        self.programs.sort(key=lambda p: p.score, reverse=True)
        self.programs = self.programs[:self.capacity]

    def best(self) -> Program:
        return self.programs[0]

    def sample_parent(self) -> Program:
        """
        Tournament selection: draw 2 candidates, return the higher scorer.
        Falls back to the single program if the island has only one member.
        """
        if len(self.programs) == 1:
            return self.programs[0]
        a, b = random.sample(self.programs, k=min(2, len(self.programs)))
        return a if a.score >= b.score else b

    def summary(self) -> str:
        scores = [f"{p.score:+.4f}" for p in self.programs]
        return (f"Island {self.island_id}  "
                f"bias=n_lags{self._init_n_lags}+{self._init_n_extra}extra  "
                f"[{', '.join(scores)}]")


# ════════════════════════════════════════════════════════════
# FUNSEARCH MAIN LOOP
# ════════════════════════════════════════════════════════════

def migrate(islands: List[Island]):
    """Copy best program from each island into every other island."""
    bests = [isl.best() for isl in islands]
    for i, island in enumerate(islands):
        for j, best in enumerate(bests):
            if j != i:
                immigrant = copy.deepcopy(best)
                immigrant.island_id = island.island_id
                island.add(immigrant)
    print("  [MIGRATION]  best scores after migration: "
          + "  ".join(f"I{isl.island_id}={isl.best().score:+.4f}"
                      for isl in islands))


def run_funsearch():
    history_rows = []
    out_history  = os.path.join(HERE, "funsearch_history.csv")
    out_best     = os.path.join(HERE, "funsearch_best.json")
    out_islands  = os.path.join(HERE, "funsearch_islands.json")

    # ── Seed: current best EpiHSD config ─────────────────
    seed = Program(program_id="seed", n_lags=5, extra_terms=[],
                   generation=0, parent_id="none")

    print("=" * 60)
    print("  FUNSEARCH  Evaluating seed program...")
    print("=" * 60)
    t0 = time.time()
    seed.score = evaluate(seed)
    print(f"  Seed score={seed.score:+.4f}  ({time.time()-t0:.1f}s)\n")

    # ── Initialise islands with biased seeds + random mutations ──
    # Each island starts from a differently shaped variant of the seed
    # (conservative, default, deep-lag, or nonlinear-heavy) so the
    # islands explore different parts of the search space from iteration 1.
    islands = [Island(i) for i in range(N_ISLANDS)]
    for isl in islands:
        # First slot: biased seed for this island
        biased = isl.make_biased_seed(seed, generation=0)
        t0 = time.time()
        biased.score = evaluate(biased)
        elapsed = time.time() - t0
        print(f"  Init I{isl.island_id} [bias-seed]  {biased.describe()}  ({elapsed:.1f}s)")
        isl.add(copy.deepcopy(seed))   # always keep the evaluated seed too
        isl.add(biased)
        history_rows.append({
            "iteration": -1, "island": isl.island_id,
            "phase": "init_bias", "program_id": biased.program_id,
            "n_lags": biased.n_lags,
            "n_extra": len(biased.extra_terms),
            "score": biased.score,
            "spec": json.dumps(biased.to_spec())})
        # Fill remaining slots by mutating the biased seed
        for _ in range(isl.capacity - 2):
            mutant = mutate(biased, generation=0)
            mutant.island_id = isl.island_id
            t0 = time.time()
            mutant.score = evaluate(mutant)
            elapsed = time.time() - t0
            print(f"  Init I{isl.island_id}  {mutant.describe()}  ({elapsed:.1f}s)")
            isl.add(mutant)
            history_rows.append({
                "iteration": -1, "island": isl.island_id,
                "phase": "init", "program_id": mutant.program_id,
                "n_lags": mutant.n_lags,
                "n_extra": len(mutant.extra_terms),
                "score": mutant.score,
                "spec": json.dumps(mutant.to_spec())})

    print()
    for isl in islands:
        print(f"  {isl.summary()}")
    print()

    # ── Main evolutionary loop ────────────────────────────
    global_best = max((isl.best() for isl in islands),
                      key=lambda p: p.score)

    for iteration in range(N_ITERATIONS):
        print(f"─── Iteration {iteration + 1:3d}/{N_ITERATIONS} "
              + "─" * 30)

        for isl in islands:
            parent = isl.sample_parent()

            # Occasionally do crossover between islands (20% chance)
            if random.random() < 0.2 and len(isl.programs) > 1:
                other = random.choice([i for i in islands if i is not isl])
                other_parent = other.sample_parent()
                child = crossover(parent, other_parent, iteration + 1)
            else:
                child = mutate(parent, iteration + 1)

            child.island_id = isl.island_id
            t0 = time.time()
            child.score = evaluate(child)
            elapsed = time.time() - t0

            isl.add(child)
            flag = " <-- NEW BEST" if child.score > isl.best().score else ""
            print(f"  I{isl.island_id}  gen={child.generation:3d}  "
                  f"{child.describe()}  ({elapsed:.1f}s){flag}")

            history_rows.append({
                "iteration": iteration + 1, "island": isl.island_id,
                "phase": "evolve", "program_id": child.program_id,
                "n_lags": child.n_lags,
                "n_extra": len(child.extra_terms),
                "score": child.score,
                "spec": json.dumps(child.to_spec())})

        # Track global best
        current_best = max((isl.best() for isl in islands),
                           key=lambda p: p.score)
        if current_best.score > global_best.score:
            global_best = copy.deepcopy(current_best)
            print(f"  *** GLOBAL BEST UPDATED  score={global_best.score:+.4f}  "
                  f"{global_best.describe()} ***")

        # Migration
        if (iteration + 1) % MIGRATE_EVERY == 0:
            migrate(islands)

        print(f"  Island bests: "
              + "  ".join(f"I{isl.island_id}={isl.best().score:+.4f}"
                          for isl in islands)
              + f"  global={global_best.score:+.4f}")
        print()

    # ── Save outputs ──────────────────────────────────────
    pd.DataFrame(history_rows).to_csv(out_history, index=False)

    with open(out_best, "w") as f:
        json.dump(global_best.to_dict(), f, indent=2)

    islands_data = {
        "global_best": global_best.to_dict(),
        "islands": [
            {"island_id": isl.island_id,
             "programs": [p.to_dict() for p in isl.programs]}
            for isl in islands
        ],
        "eval_cache_size": len(EVAL_CACHE),
        "seed_score": seed.score
    }
    with open(out_islands, "w") as f:
        json.dump(islands_data, f, indent=2)

    print("=" * 60)
    print("  FUNSEARCH COMPLETE")
    print("=" * 60)
    print(f"  Seed score      : {seed.score:+.4f}")
    print(f"  Global best     : {global_best.score:+.4f}")
    print(f"  Improvement     : {global_best.score - seed.score:+.4f}")
    print(f"  Best program    : {global_best.describe()}")
    print(f"  Evaluations run : {len(EVAL_CACHE)}")
    print(f"\n  Saved funsearch_history.csv")
    print(f"  Saved funsearch_best.json")
    print(f"  Saved funsearch_islands.json")
    print("─" * 60)

    return global_best


if __name__ == "__main__":
    best = run_funsearch()
