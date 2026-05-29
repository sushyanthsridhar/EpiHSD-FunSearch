"""
tune_n_islands.py
─────────────────────────────────────────────────────────────────────────────
Bayesian Optimization over GA SINDy hyperparameters using Optuna.

Optuna uses the TPE sampler, which is a form of Bayesian Optimization.
After each trial it builds a model of the fitness landscape and uses
it to pick the most promising hyperparameter configuration to try next.

Hyperparameters tuned
─────────────────────
  n_islands              : number of parallel islands (2 to 8)
  sa_t_initial           : SA starting temperature (0.1 to 2.0)
  sa_t_final             : SA ending temperature (0.0001 to 0.05)
  generations_per_epoch  : SA steps per island per epoch (50 to 200)
  early_stop_patience    : plateau patience in SA (20 to 80)
  max_terms              : sparsity cap on equation (5 to 15)
  r2_weight              : weight of R2 in fitness (0.5 to 0.95)
  n_llm_suggestions      : mutator variants per epoch (1 to 5)
  p_add                  : probability of add term mutation
  p_remove               : probability of remove term mutation

Fixed
─────
  p_perturb = 0.05
  p_swap    = 1 - p_add - p_remove - p_perturb

Outputs
───────
  results/bo_tuning.csv          all trial results
  results/bo_best_params.csv     best configuration found
  results/bo_tuning.log          full log

Usage
─────
  cd funsearch-ga-sindy/
  python tune_n_islands.py
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
import time
import logging
import importlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config as cfg
from run_ga_sindy import load_and_filter, build_seed_terms

# ── Tuning settings ───────────────────────────────────────────────────────────
N_TRIALS    = 30    # number of BO trials
TUNE_EPOCHS = 5     # epochs per trial (short run as proxy for full run)
N_SEEDS     = 1     # random seeds averaged per trial (1 for speed, 2 for stability)

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
LOG_PATH = os.path.join(cfg.RESULTS_DIR, "bo_tuning.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ── Config patcher ────────────────────────────────────────────────────────────

def patch_config(params: dict) -> None:
    """
    Apply trial hyperparameters to the config module at runtime.
    SA_COOLING_RATE is recomputed from the new SA_T values.
    """
    cfg.N_ISLANDS             = params["n_islands"]
    cfg.SA_T_INITIAL          = params["sa_t_initial"]
    cfg.SA_T_FINAL            = params["sa_t_final"]
    cfg.GENERATIONS_PER_EPOCH = params["generations_per_epoch"]
    cfg.EARLY_STOP_PATIENCE   = params["early_stop_patience"]
    cfg.MAX_TERMS             = params["max_terms"]
    cfg.R2_WEIGHT             = params["r2_weight"]
    cfg.SPECTRAL_WEIGHT       = round(1.0 - params["r2_weight"], 6)
    cfg.N_LLM_SUGGESTIONS     = params["n_llm_suggestions"]

    p_add    = params["p_add"]
    p_remove = params["p_remove"]
    p_swap   = max(0.01, 1.0 - p_add - p_remove - 0.05)
    # Normalize so probabilities always sum exactly to 1
    total    = p_add + p_remove + p_swap + 0.05
    cfg.MUTATION_PROBS = {
        "add_term"    : p_add    / total,
        "remove_term" : p_remove / total,
        "swap_feature": p_swap   / total,
        "crossover"   : 0.0,
        "perturb"     : 0.05     / total,
    }

    # Recompute cooling rate from new SA boundaries and step count
    cfg.SA_COOLING_RATE = (
        cfg.SA_T_FINAL / cfg.SA_T_INITIAL
    ) ** (1.0 / cfg.GENERATIONS_PER_EPOCH)


def reload_modules():
    """
    Reload modules that import config values at module level.
    Must reload in dependency order: llm_mutator -> island -> ga_coordinator.
    """
    import llm_mutator   as lmod
    import island        as imod
    import ga_coordinator as gmod

    importlib.reload(lmod)
    importlib.reload(imod)
    importlib.reload(gmod)

    return imod, gmod


# ── Single trial runner ───────────────────────────────────────────────────────

def run_trial(params: dict, province_dfs, seed_terms, seed: int) -> float:
    """
    Run one short evolution with the given hyperparameters.
    Returns the global best validation fitness achieved.
    """
    np.random.seed(seed)
    imod, gmod = reload_modules()

    islands = [
        imod.Island(
            name         = f"Island_{i+1}",
            province_dfs = province_dfs,
            seed_terms   = seed_terms,
        )
        for i in range(params["n_islands"])
    ]

    coordinator = gmod.GACoordinator(
        islands      = islands,
        province_dfs = province_dfs,
    )

    coordinator.run(verbose=False, max_epochs=TUNE_EPOCHS)
    return coordinator._global_best_fitness


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(province_dfs, seed_terms):
    """
    Return an Optuna objective function closed over the data.
    Optuna maximises the return value.
    """
    def objective(trial):
        params = {
            "n_islands"            : trial.suggest_int(   "n_islands",            2,      8),
            "sa_t_initial"         : trial.suggest_float( "sa_t_initial",         0.1,    2.0),
            "sa_t_final"           : trial.suggest_float( "sa_t_final",           0.0001, 0.05,  log=True),
            "generations_per_epoch": trial.suggest_int(   "generations_per_epoch",50,     200,   step=25),
            "early_stop_patience"  : trial.suggest_int(   "early_stop_patience",  20,     80,    step=10),
            "max_terms"            : trial.suggest_int(   "max_terms",            5,      15),
            "r2_weight"            : trial.suggest_float( "r2_weight",            0.5,    0.95),
            "n_llm_suggestions"    : trial.suggest_int(   "n_llm_suggestions",    1,      5),
            "p_add"                : trial.suggest_float( "p_add",                0.20,   0.60),
            "p_remove"             : trial.suggest_float( "p_remove",             0.10,   0.40),
        }

        patch_config(params)

        fitnesses = []
        for seed in range(N_SEEDS):
            try:
                fit = run_trial(params, province_dfs, seed_terms, seed)
                fitnesses.append(fit)
            except Exception as e:
                log.warning(f"   Trial {trial.number} seed {seed} failed: {e}")
                fitnesses.append(-1.0)

        mean_fit = float(np.mean(fitnesses))
        log.info(f"  Trial {trial.number:3d}  fitness={mean_fit:.5f}  "
                 f"n_islands={params['n_islands']}  "
                 f"T={params['sa_t_initial']:.2f}->{params['sa_t_final']:.4f}  "
                 f"gens={params['generations_per_epoch']}  "
                 f"max_terms={params['max_terms']}")
        return mean_fit

    return objective


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.error("optuna not installed. Run: pip install optuna")
        sys.exit(1)

    t_start = time.time()

    log.info("=" * 64)
    log.info("  Bayesian Optimization over GA SINDy hyperparameters")
    log.info(f"  Trials      : {N_TRIALS}")
    log.info(f"  Epochs/trial: {TUNE_EPOCHS}")
    log.info(f"  Seeds/trial : {N_SEEDS}")
    log.info("=" * 64)

    province_dfs, passing_provs, eq_lookup, sindy_r2 = load_and_filter()
    seed_terms = build_seed_terms(passing_provs, eq_lookup)

    # Create study after data loads successfully
    study = optuna.create_study(
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=42),
        study_name = "ga_sindy_bo",
    )

    log.info(f"\n  Running {N_TRIALS} trials ...\n")
    study.optimize(
        make_objective(province_dfs, seed_terms),
        n_trials = N_TRIALS,
    )

    # ── Results ───────────────────────────────────────────────────────────────
    best = study.best_trial
    log.info(f"\n{'='*64}")
    log.info("  BEST CONFIGURATION FOUND")
    log.info(f"{'='*64}")
    log.info(f"  Best fitness : {best.value:.5f}")
    log.info(f"  Trial number : {best.number}")
    for k, v in best.params.items():
        log.info(f"  {k:<28}: {v}")

    # Derived params
    p_add    = best.params["p_add"]
    p_remove = best.params["p_remove"]
    p_swap   = max(0.01, 1.0 - p_add - p_remove - 0.05)
    log.info(f"  {'p_swap (derived)':<28}: {p_swap:.3f}")
    log.info(f"  {'p_perturb (fixed)':<28}: 0.05")

    sa_cooling = (best.params["sa_t_final"] / best.params["sa_t_initial"]) ** (
        1.0 / best.params["generations_per_epoch"])
    log.info(f"  {'sa_cooling_rate (derived)':<28}: {sa_cooling:.6f}")

    # Save all trial results
    rows = []
    for tr in study.trials:
        row = {"trial": tr.number, "fitness": tr.value}
        row.update(tr.params)
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(cfg.RESULTS_DIR, "bo_tuning.csv"), index=False)

    # Save best params
    best_row = {"fitness": best.value}
    best_row.update(best.params)
    best_row["p_swap"]        = round(p_swap, 4)
    best_row["p_perturb"]     = 0.05
    best_row["sa_cooling_rate"] = round(sa_cooling, 6)
    pd.DataFrame([best_row]).to_csv(
        os.path.join(cfg.RESULTS_DIR, "bo_best_params.csv"), index=False)

    log.info(f"\n  Saved results to results/bo_tuning.csv")
    log.info(f"  Saved best params to results/bo_best_params.csv")

    elapsed = time.time() - t_start
    log.info(f"\n  Total runtime: {elapsed:.1f}s")

    log.info(f"\n  To apply best params, update config.py with:")
    log.info(f"  N_ISLANDS            = {best.params['n_islands']}")
    log.info(f"  SA_T_INITIAL         = {best.params['sa_t_initial']:.4f}")
    log.info(f"  SA_T_FINAL           = {best.params['sa_t_final']:.6f}")
    log.info(f"  GENERATIONS_PER_EPOCH= {best.params['generations_per_epoch']}")
    log.info(f"  EARLY_STOP_PATIENCE  = {best.params['early_stop_patience']}")
    log.info(f"  MAX_TERMS            = {best.params['max_terms']}")
    log.info(f"  R2_WEIGHT            = {best.params['r2_weight']:.3f}")
    log.info(f"  SPECTRAL_WEIGHT      = {round(1 - best.params['r2_weight'], 3):.3f}")
    log.info(f"  N_LLM_SUGGESTIONS    = {best.params['n_llm_suggestions']}")


if __name__ == "__main__":
    main()
