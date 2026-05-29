"""
config.py
─────────────────────────────────────────────────────────────────────────────
Single source of truth for funsearch-ga-sindy.

Key differences from funsearch/:
  - Islands no longer represent province groupings.
    Every island trains on ALL quality-passing provinces.
    Islands are parallel evolutionary searches that compete via a GA layer.
  - A GA coordinator sits above all islands and runs cross-island
    tournament selection, crossover, and migration every MIGRATION_INTERVAL
    generations.
  - The LLM mutator refines the GA-selected best equation structurally,
    then broadcasts the result as a high-quality immigrant into all islands.
  - Data quality filter is applied upfront (zero_pct < 0.40 AND
    total_cases_train >= 1000).  Only 11 provinces survive.
─────────────────────────────────────────────────────────────────────────────
"""

import os
from itertools import combinations

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE)

SEIR_CSV   = os.path.join(PROJECT_ROOT, "sindy", "seir_features.csv")
M1_SUMMARY = os.path.join(PROJECT_ROOT, "sindy", "results", "m1_summary.csv")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Target variable ───────────────────────────────────────────────────────────
TARGET_COL = "dI_dt_norm"

# ── Temporal split (chronological within each province) ───────────────────────
# 10 years total.  Split: 6 train, 2 val, 2 test.
# Adjust if your data covers different years.
TRAIN_YEARS = list(range(2014, 2020))   # 6 years
VAL_YEARS   = [2020, 2021]              # 2 years
TEST_YEARS  = [2022, 2023]              # 2 years  (held-out, never touched during evolution)

# ── Data quality filter thresholds ────────────────────────────────────────────
# Provinces failing either criterion are excluded before evolution starts.
MAX_ZERO_PCT       = 0.40   # drop if >= 40 % of training weeks are zero-case
MIN_TOTAL_CASES    = 1000   # drop if < 1000 total cases in training years

# ── Base features (10) ────────────────────────────────────────────────────────
BASE_FEATURES = [
    "incidence_lag1",
    "EIP",
    "temp_lag4",
    "rainfall_lag3",
    "avg_humidity",
    "incidence_lag2",
    "incidence_lag4",
    "rolling_incidence_4wk",
    "week_sin",
    "week_cos",
]

# ── Full candidate library (66 terms) ─────────────────────────────────────────
BIAS_TERM    = "bias"
SQUARE_TERMS = [f"{f}^2" for f in BASE_FEATURES]
CROSS_TERMS  = [f"{a} {b}" for a, b in combinations(BASE_FEATURES, 2)]
LIBRARY      = [BIAS_TERM] + BASE_FEATURES + SQUARE_TERMS + CROSS_TERMS

# ── Island model ──────────────────────────────────────────────────────────────
# Islands are independent equation searches, NOT province groupings.
# All islands train on all quality-passing provinces.
# N_ISLANDS controls diversity vs compute cost.
#   4  islands : good balance, fast runs
#   8  islands : more diversity, 2x compute
#   16 islands : maximum diversity, slow
N_ISLANDS = 6   # BO tuned from 4

# ── Within-island evolution hyperparameters ───────────────────────────────────
# All values below are BO-tuned via Bayesian Optimization (tune_n_islands.py).
GENERATIONS_PER_EPOCH  = 125   # SA steps each island runs before GA step
MAX_EPOCHS             = 20    # total epochs
EARLY_STOP_PATIENCE    = 50    # gens without best-ever improvement before early stop
MAX_TERMS              = 7     # sparsity cap. BO found sparser is better

# ── Simulated Annealing parameters (within each island) ──────────────────────
# Temperature controls how often worse equations are accepted.
# High T  -> accepts almost anything (broad exploration).
# Low  T  -> only accepts improvements (exploitation / convergence).
# Cooling schedule: T(g) = SA_T_INITIAL * SA_COOLING_RATE^g
SA_T_INITIAL    = 1.1828    # BO tuned. higher start = more initial exploration
SA_T_FINAL      = 0.000377  # BO tuned. lower end = tighter convergence
SA_COOLING_RATE = 0.937614  # derived from BO best params

# Legacy population size kept for backwards compat with any external references
POPULATION_SIZE = 1
TOURNAMENT_K    = 3

# ── GA coordinator hyperparameters ───────────────────────────────────────────
# After every epoch:
#   1. Collect island champions (one best equation per island).
#   2. Tournament-select the top 2 champions.
#   3. Crossover them to produce an offspring.
#   4. Send offspring to LLM mutator for structural refinement.
#   5. Inject LLM-refined equation as immigrant into ALL islands.
GA_TOURNAMENT_K        = 2     # number of island champions compared per GA step
N_LLM_SUGGESTIONS      = 3     # how many mutator variants to generate per epoch
MIGRATION_TOP_N        = 1     # how many migrants to inject per island per epoch

# ── Mutation operator probabilities (within each island) ─────────────────────
# BO-tuned. swap is now dominant. add and remove share the rest evenly.
MUTATION_PROBS = {
    "add_term"    : 0.326,
    "remove_term" : 0.218,
    "swap_feature": 0.406,
    "crossover"   : 0.0,
    "perturb"     : 0.05,
}

# ── Numeric constants ─────────────────────────────────────────────────────────
GRADIENT_STEP  = 0.02
GRADIENT_LR    = 0.03
PERTURB_SIGMA  = 0.10
NEW_TERM_SIGMA = 0.20

# ── Fitness weights ───────────────────────────────────────────────────────────
# BO found spectral score matters much more than originally assumed.
R2_WEIGHT       = 0.510    # BO tuned from 0.7
SPECTRAL_WEIGHT = 0.490    # BO tuned from 0.3
N_SPECTRAL_FREQS = 5

# ── LLM mutator mode ──────────────────────────────────────────────────────────
# "algorithmic" : use the same add/remove/swap operators as before (default, no API).
# "claude"      : call the Anthropic Claude API for structural suggestions.
#                 Requires ANTHROPIC_API_KEY environment variable.
LLM_MODE = "algorithmic"

# ── Max-islands calculator (informational, printed at startup) ────────────────
def compute_max_islands(n_provinces, train_years, weeks_per_year=52,
                        min_rows_per_term=10, max_terms=MAX_TERMS):
    """
    Compute the maximum sensible number of islands given the data budget.

    Since all islands share all province data, the limit is not about data
    scarcity but about having enough population diversity.  We report both
    the data-theoretic maximum (essentially unlimited when all islands share
    data) and a practical recommendation based on compute cost.
    """
    total_train_rows = n_provinces * len(train_years) * weeks_per_year
    min_rows_needed  = max_terms * min_rows_per_term
    data_max         = total_train_rows // min_rows_needed
    practical_max    = min(16, n_provinces // 2)
    return {
        "total_train_rows"   : total_train_rows,
        "min_rows_needed_ols": min_rows_needed,
        "data_theoretical_max": data_max,
        "practical_recommended_max": practical_max,
    }
