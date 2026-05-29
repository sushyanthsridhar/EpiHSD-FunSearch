"""
sindy_mutator.py
────────────────────────────────────────────────────────────────
Structural library mutator for SINDy. No LLM required.

Mimics what FunSearch will do with an LLM, but uses residual-
guided search instead. Two components talk to each other:

  SINDyIsland   fits equations on a given library, returns
                coefficients + residuals + val R2.

  LibraryMutator receives those results, inspects which
                equations are weak, correlates their residuals
                against all candidate features, and proposes
                the next library to try.

EXCHANGE LOOP (N_ROUNDS):
  Mutator --> SINDy : library L_k
  SINDy   --> Mutator : Xi_k, residuals_k, val_r2_k
  Mutator decides L_{k+1} based on what it sees
  repeat.

Candidate pool:
  Polynomial z terms (degree 1-2) + annual external forcings.

TEST DATA (2022-2023) IS LOCKED until the very end.

Run:
  python3 sindy/sindy_mutator.py
────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ── Config ─────────────────────────────────────────────────────
LATENT_DIM     = 3
POLY_DEG       = 2        # polynomial z terms in base library
STLSQ_THRESH   = 0.05     # inner STLSQ sparsity threshold
N_ROUNDS       = 40       # exchange rounds
N_MUTATIONS    = 15       # candidate libraries per round
TOP_K          = 5        # elite libraries carried forward
LAMBDA_TERMS   = 0.01     # sparsity penalty per active term in fitness
R2_WEIGHTS     = [0.6, 0.2, 0.2]  # weight z1, z2, z3 in fitness
                                   # z1 weighted highest as it is the problem axis

TRAIN_YEARS      = list(range(2015, 2020))
VAL_YEARS        = [2020, 2021]
TEST_YEARS       = [2022, 2023]   # LOCKED

# External candidate features (annual-level per province)
EXTERNAL_POOL = [
    "nino34_anom", "soi_index",
    "neighbor_incidence_lag1", "neighbor_incidence_lag2",
    "avg_rainfall", "max_rainfall", "avg_temp",
    "consecutive_dry_weeks", "consecutive_wet_weeks",
    "avg_humidity", "solar_radiation",
]

z_cols = [f"z{i+1}" for i in range(LATENT_DIM)]

# ── Load data ───────────────────────────────────────────────────
latent_tv  = pd.read_csv(os.path.join(ROOT, "autoencoder",
                          f"latent_representations_latent{LATENT_DIM}.csv"))
latent_all = pd.read_csv(os.path.join(ROOT, "autoencoder",
                          f"latent_all_years_latent{LATENT_DIM}.csv"))

ext     = pd.read_csv(os.path.join(ROOT, "extended_input_normalized.csv"))
id_cols = ["province", "year", "month", "week"]
annual  = ext.groupby(["province", "year"])[
    [c for c in ext.columns if c not in id_cols]
].mean().reset_index()

# Filter external pool to columns that exist
EXTERNAL_POOL = [c for c in EXTERNAL_POOL if c in annual.columns]
print(f"External pool ({len(EXTERNAL_POOL)}): {EXTERNAL_POOL}")

def merge_ext(latent_df):
    return latent_df.merge(
        annual[["province", "year"] + EXTERNAL_POOL],
        on=["province", "year"], how="left"
    )

tv_df  = merge_ext(latent_tv)
all_df = merge_ext(latent_all)

# ── Build (state, derivative, external) pairs ───────────────────
def build_pairs(df, year_transition_filter=None):
    """Return Z (state), dZ (derivative), E (external at time t)."""
    Z_list, dZ_list, E_list = [], [], []
    for province, grp in df.groupby("province"):
        grp = grp.sort_values("year").reset_index(drop=True)
        for t in range(len(grp) - 1):
            yr_t = grp.loc[t, "year"]
            if grp.loc[t + 1, "year"] - yr_t != 1:
                continue
            if year_transition_filter and yr_t not in year_transition_filter:
                continue
            Z_list.append(grp.loc[t, z_cols].values.astype(float))
            dZ_list.append(
                grp.loc[t + 1, z_cols].values.astype(float)
                - grp.loc[t, z_cols].values.astype(float)
            )
            E_list.append(grp.loc[t, EXTERNAL_POOL].values.astype(float))
    return np.array(Z_list), np.array(dZ_list), np.array(E_list)

# Train: transitions 2015->16, 16->17, 17->18, 18->19
Z_tr, dZ_tr, E_tr = build_pairs(tv_df, year_transition_filter=[2015,2016,2017,2018])
# Val:   transitions 2019->20, 2020->21
Z_va, dZ_va, E_va = build_pairs(tv_df, year_transition_filter=[2019, 2020])
print(f"Train pairs: {len(Z_tr)}   Val pairs: {len(Z_va)}")

# ── Polynomial base library ─────────────────────────────────────
poly  = PolynomialFeatures(degree=POLY_DEG, include_bias=True)
poly.fit(Z_tr)
z_term_names  = list(poly.get_feature_names_out(z_cols))
ALL_TERMS     = z_term_names + EXTERNAL_POOL
N_TERMS       = len(ALL_TERMS)
N_Z_TERMS     = len(z_term_names)
print(f"Total candidate terms: {N_TERMS}  ({N_Z_TERMS} poly-z + {len(EXTERNAL_POOL)} external)")

def build_theta(Z, E, term_mask=None):
    """Build feature matrix. term_mask selects which columns to include."""
    Tz = poly.transform(Z)                         # (N, n_z_terms)
    T  = np.hstack([Tz, E])                        # (N, N_TERMS)
    if term_mask is not None:
        return T[:, term_mask]
    return T

Theta_tr = build_theta(Z_tr, E_tr)
Theta_va = build_theta(Z_va, E_va)

# ── STLSQ ───────────────────────────────────────────────────────
def stlsq(Theta, dZ, threshold, max_iter=25):
    Xi = np.linalg.lstsq(Theta, dZ, rcond=None)[0]
    for _ in range(max_iter):
        prev = Xi.copy()
        for j in range(dZ.shape[1]):
            active = np.abs(Xi[:, j]) >= threshold
            Xi[:, j] = 0.0
            if active.sum() > 0:
                Xi[active, j] = np.linalg.lstsq(
                    Theta[:, active], dZ[:, j], rcond=None)[0]
        if np.allclose(Xi, prev, atol=1e-12):
            break
    return Xi

# ── SINDy Island ────────────────────────────────────────────────
class SINDyIsland:
    """
    Fits SINDy on train given a library (term_mask).
    Returns coefficients, residuals, and val R2 back to the mutator.
    """
    def __init__(self, Theta_tr, dZ_tr, Theta_va, dZ_va):
        self.Theta_tr = Theta_tr
        self.dZ_tr    = dZ_tr
        self.Theta_va = Theta_va
        self.dZ_va    = dZ_va

    def run(self, term_mask):
        """
        term_mask: bool array of length N_TERMS
        Returns dict with: Xi_full, residuals_tr, val_r2, train_r2
        """
        Xi_full = np.zeros((N_TERMS, LATENT_DIM))
        active_idx = np.where(term_mask)[0]
        if len(active_idx) == 0:
            return self._empty_result()

        Xi_sub = stlsq(
            self.Theta_tr[:, active_idx],
            self.dZ_tr,
            STLSQ_THRESH,
        )
        Xi_full[active_idx] = Xi_sub

        dZ_tr_pred = self.Theta_tr[:, active_idx] @ Xi_sub
        dZ_va_pred = self.Theta_va[:, active_idx] @ Xi_sub

        residuals_tr = self.dZ_tr - dZ_tr_pred

        train_r2, val_r2 = [], []
        for j in range(LATENT_DIM):
            def r2(true, pred):
                ss_r = ((true - pred) ** 2).sum()
                ss_t = ((true - true.mean()) ** 2).sum()
                return float(1 - ss_r / (ss_t + 1e-10))
            train_r2.append(r2(self.dZ_tr[:, j], dZ_tr_pred[:, j]))
            val_r2.append(r2(self.dZ_va[:, j], dZ_va_pred[:, j]))

        return {
            "Xi_full":      Xi_full,
            "residuals_tr": residuals_tr,
            "train_r2":     train_r2,
            "val_r2":       val_r2,
            "n_active":     int((np.abs(Xi_full) > 1e-12).sum()),
        }

    def _empty_result(self):
        return {
            "Xi_full":      np.zeros((N_TERMS, LATENT_DIM)),
            "residuals_tr": self.dZ_tr.copy(),
            "train_r2":     [-999.0] * LATENT_DIM,
            "val_r2":       [-999.0] * LATENT_DIM,
            "n_active":     0,
        }

# ── Library Mutator ─────────────────────────────────────────────
class LibraryMutator:
    """
    Receives SINDy results and proposes the next library to try.

    Mutation strategies:
      residual_add    : find the external feature most correlated with
                        the worst equation residuals and add it.
      coef_prune      : remove the term with the smallest absolute coef.
      random_add      : add a random unused term.
      random_remove   : remove a random active term.
      swap            : remove weakest active, add best residual-correlated.
      elite_crossover : combine two top-K libraries.
    """
    def __init__(self, elite_pool):
        # elite_pool: list of (term_mask, fitness, sindy_result)
        self.elite = elite_pool
        self.rng   = np.random.default_rng(seed=0)

    def propose_mutations(self, n):
        proposals = []
        for _ in range(n):
            parent_mask, _, parent_result = self.elite[
                self.rng.integers(0, len(self.elite))
            ]
            strategy = self.rng.choice(
                ["residual_add", "coef_prune", "random_add",
                 "random_remove", "swap", "elite_crossover"],
                p=[0.30, 0.15, 0.15, 0.10, 0.20, 0.10],
            )
            child = self._apply(parent_mask, parent_result, strategy)
            proposals.append(child)
        return proposals

    def _apply(self, mask, result, strategy):
        m = mask.copy()
        val_r2  = np.array(result["val_r2"])
        Xi_full = result["Xi_full"]
        resid   = result["residuals_tr"]   # (N_pairs, D)

        if strategy == "residual_add":
            # For the weakest equation, find the external feature
            # most correlated with its residuals.
            worst_j = int(np.argmin(val_r2))
            res_j   = resid[:, worst_j]
            ext_corr = []
            for ei, name in enumerate(EXTERNAL_POOL):
                col_idx  = N_Z_TERMS + ei
                if m[col_idx]:
                    ext_corr.append(-1.0)   # already active
                    continue
                ext_vals = E_tr[:, ei]
                if ext_vals.std() < 1e-8:
                    ext_corr.append(0.0)
                    continue
                c = float(np.corrcoef(res_j, ext_vals)[0, 1])
                ext_corr.append(abs(c) if not np.isnan(c) else 0.0)
            best_ext = int(np.argmax(ext_corr))
            m[N_Z_TERMS + best_ext] = True

        elif strategy == "coef_prune":
            # Remove the active term with the smallest total |coef|
            total_mag = np.abs(Xi_full).sum(axis=1)
            active    = np.where(m)[0]
            if len(active) > 1:
                weakest = active[np.argmin(total_mag[active])]
                m[weakest] = False

        elif strategy == "random_add":
            inactive = np.where(~m)[0]
            if len(inactive) > 0:
                m[self.rng.choice(inactive)] = True

        elif strategy == "random_remove":
            active = np.where(m)[0]
            # never remove bias (index 0)
            removable = active[active > 0]
            if len(removable) > 0:
                m[self.rng.choice(removable)] = False

        elif strategy == "swap":
            # Remove weakest active term, add best residual-correlated external
            worst_j = int(np.argmin(val_r2))
            res_j   = resid[:, worst_j]
            active  = np.where(m)[0]
            removable = active[active > 0]
            if len(removable) > 0:
                total_mag = np.abs(Xi_full).sum(axis=1)
                weakest   = removable[np.argmin(total_mag[removable])]
                m[weakest] = False
            # add best uncorrelated external
            ext_corr = []
            for ei in range(len(EXTERNAL_POOL)):
                col_idx = N_Z_TERMS + ei
                if m[col_idx]:
                    ext_corr.append(-1.0)
                    continue
                ext_vals = E_tr[:, ei]
                if ext_vals.std() < 1e-8:
                    ext_corr.append(0.0)
                    continue
                c = float(np.corrcoef(res_j, ext_vals)[0, 1])
                ext_corr.append(abs(c) if not np.isnan(c) else 0.0)
            best_ext = int(np.argmax(ext_corr))
            m[N_Z_TERMS + best_ext] = True

        elif strategy == "elite_crossover":
            if len(self.elite) >= 2:
                other_mask = self.elite[self.rng.integers(1, len(self.elite))][0]
                # take union of active terms, then randomly drop half the non-shared
                shared   = m & other_mask
                only_a   = m & ~other_mask
                only_b   = ~m & other_mask
                m = shared.copy()
                for idx in np.where(only_a)[0]:
                    if self.rng.random() > 0.5:
                        m[idx] = True
                for idx in np.where(only_b)[0]:
                    if self.rng.random() > 0.5:
                        m[idx] = True

        # always keep bias term
        m[0] = True
        return m

# ── Fitness ─────────────────────────────────────────────────────
def fitness(result):
    val_r2  = result["val_r2"]
    n_act   = result["n_active"]
    weighted = sum(R2_WEIGHTS[j] * val_r2[j] for j in range(LATENT_DIM))
    return weighted - LAMBDA_TERMS * n_act

# ── Initialize population ────────────────────────────────────────
island  = SINDyIsland(Theta_tr, dZ_tr, Theta_va, dZ_va)
rng     = np.random.default_rng(42)

# Seed 1: all z polynomial terms only
seed_mask_poly = np.array([True] * N_Z_TERMS + [False] * len(EXTERNAL_POOL))
# Seed 2: all z + all external
seed_mask_all  = np.ones(N_TERMS, dtype=bool)
# Seed 3-10: random
seeds = [seed_mask_poly, seed_mask_all]
for _ in range(8):
    m = np.zeros(N_TERMS, dtype=bool)
    m[0] = True
    m[rng.integers(1, N_Z_TERMS, size=4)] = True
    for ei in range(len(EXTERNAL_POOL)):
        if rng.random() < 0.4:
            m[N_Z_TERMS + ei] = True
    seeds.append(m)

population = []
for m in seeds:
    res = island.run(m)
    f   = fitness(res)
    population.append((m, f, res))
population.sort(key=lambda x: x[1], reverse=True)
population = population[:TOP_K]

# ── Exchange loop ────────────────────────────────────────────────
history = []
print(f"\nStarting exchange loop: {N_ROUNDS} rounds, {N_MUTATIONS} mutations/round")
print(f"  Fitness = 0.6*val_r2_z1 + 0.2*val_r2_z2 + 0.2*val_r2_z3 - {LAMBDA_TERMS}*n_terms")

for round_idx in range(N_ROUNDS):
    mutator    = LibraryMutator(population)
    proposals  = mutator.propose_mutations(N_MUTATIONS)

    new_entries = []
    for prop_mask in proposals:
        res = island.run(prop_mask)
        f   = fitness(res)
        new_entries.append((prop_mask, f, res))

    combined = population + new_entries
    combined.sort(key=lambda x: x[1], reverse=True)
    population = combined[:TOP_K]

    best_f   = population[0][1]
    best_res = population[0][2]
    history.append({
        "round":    round_idx + 1,
        "fitness":  best_f,
        "val_r2_z1": best_res["val_r2"][0],
        "val_r2_z2": best_res["val_r2"][1],
        "val_r2_z3": best_res["val_r2"][2],
        "n_active": best_res["n_active"],
    })

    if (round_idx + 1) % 10 == 0:
        print(f"  Round {round_idx+1:3d}  fitness={best_f:.4f}  "
              f"val_r2=[{best_res['val_r2'][0]:.3f}, "
              f"{best_res['val_r2'][1]:.3f}, "
              f"{best_res['val_r2'][2]:.3f}]  "
              f"n_terms={best_res['n_active']}")

# ── Best library results ─────────────────────────────────────────
best_mask, best_fit, best_res = population[0]
Xi_best = best_res["Xi_full"]

print("\n" + "=" * 68)
print("  BEST LIBRARY FOUND BY MUTATOR")
print("=" * 68)
active_terms = [ALL_TERMS[i] for i in range(N_TERMS) if best_mask[i]]
ext_used     = [t for t in active_terms if t in EXTERNAL_POOL]
z_used       = [t for t in active_terms if t in z_term_names]
print(f"  Z polynomial terms : {z_used}")
print(f"  External forcings  : {ext_used}")
print(f"  Total active terms : {len(active_terms)}")

print(f"\n  Fitness = {best_fit:.4f}")
print(f"  Val R2  : z1={best_res['val_r2'][0]:.4f}  "
      f"z2={best_res['val_r2'][1]:.4f}  z3={best_res['val_r2'][2]:.4f}")
print(f"  Train R2: z1={best_res['train_r2'][0]:.4f}  "
      f"z2={best_res['train_r2'][1]:.4f}  z3={best_res['train_r2'][2]:.4f}")

print("\n  EQUATIONS:")
for j, zname in enumerate(z_cols):
    terms = [(ALL_TERMS[i], Xi_best[i, j])
             for i in range(N_TERMS) if abs(Xi_best[i, j]) > 1e-12]
    print(f"\n  d{zname}/dt =")
    for tname, coef in terms:
        src = "[EXT]" if tname in EXTERNAL_POOL else "     "
        print(f"    {src}  {coef:+.5f}  *  {tname}")
    if not terms:
        print("    0  (stable axis)")
print("=" * 68)

# ── Test rollout with best library ───────────────────────────────
print("\n  TEST ROLLOUT (2022-2023, locked data):")
all_actual, all_pred = [], []
for province in sorted(all_df["province"].unique()):
    sub  = all_df[all_df["province"] == province].sort_values("year")
    seed = sub[sub["year"] == 2021]
    if seed.empty:
        continue
    z_cur = seed[z_cols].values[0]
    test  = sub[sub["year"].isin(TEST_YEARS)].sort_values("year")
    if test.empty:
        continue
    preds = []
    for _, row in test.iterrows():
        e_row   = row[EXTERNAL_POOL].values.astype(float).reshape(1, -1)
        t_row   = build_theta(z_cur.reshape(1, -1), e_row)
        dz      = (t_row @ Xi_best).flatten()
        z_cur   = z_cur + dz
        preds.append(z_cur.copy())
    all_actual.append(test[z_cols].values)
    all_pred.append(np.array(preds))

all_actual = np.vstack(all_actual)
all_pred   = np.vstack(all_pred)

for j, zname in enumerate(z_cols):
    ss_res = ((all_actual[:, j] - all_pred[:, j]) ** 2).sum()
    ss_tot = ((all_actual[:, j] - all_actual[:, j].mean()) ** 2).sum()
    r2  = 1 - ss_res / (ss_tot + 1e-10)
    mae = np.abs(all_actual[:, j] - all_pred[:, j]).mean()
    print(f"    {zname}  R2={r2:.4f}  MAE={mae:.4f}")

# ── Save outputs ─────────────────────────────────────────────────
hist_df = pd.DataFrame(history)
hist_df.to_csv(os.path.join(OUT_DIR := HERE,
               f"mutator_history_latent{LATENT_DIM}.csv"), index=False)

coef_df = pd.DataFrame(Xi_best, index=ALL_TERMS,
                        columns=[f"d{z}/dt" for z in z_cols])
coef_df.to_csv(os.path.join(OUT_DIR,
               f"mutator_best_coefficients_latent{LATENT_DIM}.csv"))

# ── Plots ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(hist_df["round"], hist_df["fitness"], "k-", lw=2, label="fitness")
ax.set_xlabel("Round")
ax.set_ylabel("Fitness")
ax.set_title("Mutator Convergence")
ax.grid(True, alpha=0.3)

ax = axes[1]
for j, zname in enumerate(z_cols):
    ax.plot(hist_df["round"], hist_df[f"val_r2_{zname}"],
            label=f"val R2 {zname}", lw=2)
ax.axhline(0, color="gray", lw=1, linestyle="--")
ax.set_xlabel("Round")
ax.set_ylabel("Val R2")
ax.set_title("Val R2 per Equation")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,
            f"mutator_convergence_latent{LATENT_DIM}.png"), dpi=150)
plt.close()
print(f"\nSaved mutator_convergence_latent{LATENT_DIM}.png")
print(f"Saved mutator_history_latent{LATENT_DIM}.csv")
print(f"Saved mutator_best_coefficients_latent{LATENT_DIM}.csv")
print("\n── DONE ─────────────────────────────────────────────────")
