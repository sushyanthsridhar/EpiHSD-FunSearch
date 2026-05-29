"""
config.py
─────────────────────────────────────────────────────────────────────────────
Single source of truth for FunSearch-SINDy (Project 2).

All other modules import from here — nothing is hardcoded elsewhere.
─────────────────────────────────────────────────────────────────────────────
"""

import os
from itertools import combinations

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE          = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(BASE)

SEIR_CSV      = os.path.join(PROJECT_ROOT, "sindy", "seir_features.csv")
M1_SUMMARY    = os.path.join(PROJECT_ROOT, "sindy", "results", "m1_summary.csv")
RESULTS_DIR   = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Target variable ───────────────────────────────────────────────────────────
TARGET_COL = "dI_dt_norm"

# ── Temporal split ────────────────────────────────────────────────────────────
# Fixed single split — 5 years train, 2 years val, 2 years test.
# (Replaces the old 4-window walk-forward to give the model more training data.)
TRAIN_YEARS = list(range(2015, 2020))   # 2015–2019  (5 years, 260 rows/province)
VAL_YEARS   = [2020, 2021]              # 2020–2021  (2 years, 104 rows/province)
TEST_YEARS  = [2022, 2023]              # 2022–2023  (2 years, never touched until eval)

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
# Degree-0: bias / intercept
# Degree-1: 10 base features
# Degree-2: 10 squares + 45 cross-products = 55 interaction terms
# Total: 1 + 10 + 55 = 66

BIAS_TERM = "bias"

SQUARE_TERMS = [f"{f}^2" for f in BASE_FEATURES]

CROSS_TERMS  = [f"{a} {b}" for a, b in combinations(BASE_FEATURES, 2)]

LIBRARY = [BIAS_TERM] + BASE_FEATURES + SQUARE_TERMS + CROSS_TERMS
# len(LIBRARY) == 66

# ── Province list (canonical names matching seir_features.csv) ────────────────
ALL_PROVINCES = [
    "32 Santo Domingo",
    "25 Santiago",
    "01 Distrito Nacional",
    "21 San Cristóbal",
    "13 La Vega",
    "04 Barahona",
    "06 Duarte",
    "18 Puerto Plata",
    "11 La Altagracia",
    "22 San Juan",
    "15 Monte Cristi",
    "20 Samaná",
    "30 Hato Mayor",
    "29 Monte Plata",
    "16 Pedernales",
    "28 Monseñor Nouel",
    "07 Elías Piña",
    "24 Sánchez Ramírez",
    "31 San José de Ocoa",
    "03 Baoruco",
    "02 Azua",
    "05 Dajabón",
    "08 El Seibo",
    "09 Espaillat",
    "10 Independencia",
    "12 La Romana",
    "14 María Trinidad Sánchez",
    "17 Peravia",
    "19 Hermanas Mirabal",
    "23 San Pedro de Macorís",
    "26 Santiago Rodríguez",
    "27 Valverde",
]

# ── Island mode ───────────────────────────────────────────────────────────────
# "32"  — one island per province (current default)
# "2"   — two islands: coastal EIP-dominant vs inland AR-dominant
# Change this one value to switch modes everywhere in the codebase.
ISLAND_MODE = "2"

# ── Island definitions ────────────────────────────────────────────────────────
# ISLANDS is a list of province lists. Each sub-list is one island.
# seed_loader.py and the evolution loop consume this directly.

if ISLAND_MODE == "32":
    # One island per province — 32 independent evolutionary searches
    ISLANDS = [[p] for p in ALL_PROVINCES]

elif ISLAND_MODE == "2":
    # Biologically motivated grouping from SINDy Stage 6 analysis
    _COASTAL = [
        "32 Santo Domingo", "01 Distrito Nacional", "13 La Vega",
        "04 Barahona", "06 Duarte", "24 Sánchez Ramírez",
        "11 La Altagracia", "22 San Juan", "28 Monseñor Nouel",
        "31 San José de Ocoa", "30 Hato Mayor", "20 Samaná", "03 Baoruco",
    ]
    _INLAND = [p for p in ALL_PROVINCES if p not in _COASTAL]
    ISLANDS = [_COASTAL, _INLAND]

else:
    raise ValueError(f"Unknown ISLAND_MODE: {ISLAND_MODE!r}. Use '32' or '2'.")

N_ISLANDS = len(ISLANDS)

# ── Training windows ──────────────────────────────────────────────────────────
# Single fixed split — all evolution happens here.
# val and test are both lists so _evaluate() can average over multiple years.
WINDOWS = [
    {
        "train" : list(range(2015, 2020)),   # 2015–2019
        "val"   : [2020, 2021],              # 2020–2021
        "test"  : [2022, 2023],              # 2022–2023  (held-out)
    }
]
N_WINDOWS = len(WINDOWS)   # 1

# ── Evolution hyperparameters ──────────────────────────────────────────────────
POPULATION_SIZE         = 20    # equations per island
GENERATIONS_PER_WINDOW  = 200   # mutation rounds (increased: single window, more budget)
TOURNAMENT_K            = 3     # tournament selection size
EARLY_STOP_DELTA        = 0.005 # minimum R² improvement to count as progress
EARLY_STOP_PATIENCE     = 50    # generations without improvement → stop early
MAX_TERMS               = 10    # force remove_term if equation exceeds this

# ── Mutation probabilities ─────────────────────────────────────────────────────
# Structural operators dominate — coefficients are re-fitted by SINDy each step.
# perturb/gradient_perturb kept at low weight as a fallback exploration tool.
MUTATION_PROBS = {
    "add_term"         : 0.35,   # add a random library term
    "remove_term"      : 0.25,   # drop the weakest term (sparsity pressure)
    "swap_feature"     : 0.25,   # replace one term with another
    "crossover"        : 0.10,   # recombine two parent structures
    "perturb"          : 0.05,   # small coefficient nudge (rarely used)
}

GRADIENT_STEP   = 0.02
GRADIENT_LR     = 0.03

PERTURB_SIGMA      = 0.10
NEW_TERM_SIGMA     = 0.20
