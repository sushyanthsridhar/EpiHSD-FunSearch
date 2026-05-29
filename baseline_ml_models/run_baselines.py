"""
run_baselines.py
────────────────────────────────────────────────────────────────
Baseline ML models for dengue incidence forecasting.

No autoencoder, no latent space. Raw features only.

INPUT
─────
  ../data_processing/input.csv
  Columns: province, year, month, week, total_cases,
           avg_temp, max_temp, min_temp, avg_rainfall,
           max_rainfall, min_rainfall, avg_humidity,
           max_humidity, min_humidity, population

FEATURE ENGINEERING
───────────────────
  Lag features: total_cases at t-1..t-N_LAGS (within province)
  Circular time encoding: sin/cos of week number
  All raw climate and demographic features at time t
  Target: total_cases at t+1 (next week, within province)

SPLITS (same as EpiHSD pipeline)
─────────────────────────────────
  Train : 2015-2019
  Val   : 2020-2021  (used for hyperparam tuning)
  Test  : 2022-2023  (sealed, evaluated once at end)

MODELS
──────
  1. Linear Regression (no regularisation)
  2. Ridge Regression  (alpha tuned)
  3. Lasso Regression  (alpha tuned)
  4. Random Forest     (n_estimators, max_depth, min_samples_leaf tuned)
  5. Gradient Boosting (n_estimators, learning_rate, max_depth tuned)
  6. SVR               (C, epsilon, kernel tuned)

OUTPUTS
───────
  baseline_results.csv     val and test R2 + RMSE for every model
  baseline_predictions.csv province-week predictions on test set
────────────────────────────────────────────────────────────────
"""

import os
import warnings
import numpy as np
import pandas as pd
from itertools import product
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, "..", "data_processing", "input.csv")

TRAIN_YEARS = list(range(2015, 2020))
VAL_YEARS   = [2020, 2021]
TEST_YEARS  = [2022, 2023]
N_LAGS      = 5      # lag weeks for incidence history

# ── Load and sort ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_CSV)
df = df.sort_values(["province", "year", "week"]).reset_index(drop=True)

FEATURE_COLS = [
    "avg_temp", "max_temp", "min_temp",
    "avg_rainfall", "max_rainfall", "min_rainfall",
    "avg_humidity", "max_humidity", "min_humidity",
    "population"
]

# ── Feature engineering (within province) ─────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prov, grp in df.groupby("province"):
        grp = grp.sort_values(["year", "week"]).reset_index(drop=True)
        cases = grp["total_cases"].values
        T = len(grp)
        for t in range(N_LAGS, T - 1):
            row = {}
            row["province"] = prov
            row["year"]     = grp.loc[t, "year"]
            row["week"]     = grp.loc[t, "week"]
            # circular time encoding
            w = grp.loc[t, "week"]
            row["time_sin"] = np.sin(2 * np.pi * w / 52)
            row["time_cos"] = np.cos(2 * np.pi * w / 52)
            # lag features
            for lag in range(1, N_LAGS + 1):
                row[f"cases_lag{lag}"] = cases[t - lag + 1]
            # climate and demographic features at time t
            for col in FEATURE_COLS:
                row[col] = grp.loc[t, col]
            # target: next week incidence
            row["target"] = cases[t + 1]
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)

print("Building features...")
feat_df = build_features(df)

id_cols  = ["province", "year", "week", "target"]
X_cols   = [c for c in feat_df.columns if c not in id_cols]

train_mask = feat_df["year"].isin(TRAIN_YEARS)
val_mask   = feat_df["year"].isin(VAL_YEARS)
test_mask  = feat_df["year"].isin(TEST_YEARS)

X_train = feat_df.loc[train_mask, X_cols].values.astype(np.float32)
y_train = feat_df.loc[train_mask, "target"].values.astype(np.float32)
X_val   = feat_df.loc[val_mask,   X_cols].values.astype(np.float32)
y_val   = feat_df.loc[val_mask,   "target"].values.astype(np.float32)
X_test  = feat_df.loc[test_mask,  X_cols].values.astype(np.float32)
y_test  = feat_df.loc[test_mask,  "target"].values.astype(np.float32)

print(f"  Train: {X_train.shape[0]} samples")
print(f"  Val  : {X_val.shape[0]} samples")
print(f"  Test : {X_test.shape[0]} samples")
print(f"  Features: {X_train.shape[1]}\n")

# Scale for linear models and SVR
scaler  = StandardScaler()
Xs_tr   = scaler.fit_transform(X_train)
Xs_val  = scaler.transform(X_val)
Xs_test = scaler.transform(X_test)

# ── Helpers ────────────────────────────────────────────────────────────────────
def score(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse

def grid_search(model_fn, param_grid, Xtr, ytr, Xv, yv):
    """Return best model and best val R2 over a manual grid."""
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    best_r2, best_model, best_params = -np.inf, None, None
    for combo in product(*values):
        params = dict(zip(keys, combo))
        m = model_fn(**params)
        m.fit(Xtr, ytr)
        r2, _ = score(yv, m.predict(Xv))
        if r2 > best_r2:
            best_r2, best_model, best_params = r2, m, params
    return best_model, best_params, best_r2

# ── Model definitions and grids ───────────────────────────────────────────────
results = []
test_preds_all = feat_df.loc[test_mask, ["province", "year", "week"]].copy()

# 1. Linear Regression
print("Fitting Linear Regression...")
lr = LinearRegression()
lr.fit(Xs_tr, y_train)
v_r2, v_rmse = score(y_val,  lr.predict(Xs_val))
t_r2, t_rmse = score(y_test, lr.predict(Xs_test))
results.append({"model": "Linear Regression", "val_r2": v_r2, "val_rmse": v_rmse,
                "test_r2": t_r2, "test_rmse": t_rmse, "best_params": {}})
test_preds_all["Linear Regression"] = lr.predict(Xs_test)
print(f"  val R2={v_r2:.4f}  test R2={t_r2:.4f}")

# 2. Ridge
print("Fitting Ridge...")
ridge_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
best_ridge, bp, _ = grid_search(Ridge, ridge_grid, Xs_tr, y_train, Xs_val, y_val)
v_r2, v_rmse = score(y_val,  best_ridge.predict(Xs_val))
t_r2, t_rmse = score(y_test, best_ridge.predict(Xs_test))
results.append({"model": "Ridge", "val_r2": v_r2, "val_rmse": v_rmse,
                "test_r2": t_r2, "test_rmse": t_rmse, "best_params": bp})
test_preds_all["Ridge"] = best_ridge.predict(Xs_test)
print(f"  val R2={v_r2:.4f}  test R2={t_r2:.4f}  params={bp}")

# 3. Lasso
print("Fitting Lasso...")
lasso_grid = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
best_lasso, bp, _ = grid_search(Lasso, lasso_grid, Xs_tr, y_train, Xs_val, y_val)
v_r2, v_rmse = score(y_val,  best_lasso.predict(Xs_val))
t_r2, t_rmse = score(y_test, best_lasso.predict(Xs_test))
results.append({"model": "Lasso", "val_r2": v_r2, "val_rmse": v_rmse,
                "test_r2": t_r2, "test_rmse": t_rmse, "best_params": bp})
test_preds_all["Lasso"] = best_lasso.predict(Xs_test)
print(f"  val R2={v_r2:.4f}  test R2={t_r2:.4f}  params={bp}")

# 4. Random Forest
print("Fitting Random Forest (grid search)...")
rf_grid = {
    "n_estimators":    [200],
    "max_depth":       [None, 10, 20],
    "min_samples_leaf":[1, 5],
    "random_state":    [42]
}
best_rf, bp, _ = grid_search(RandomForestRegressor, rf_grid, X_train, y_train, X_val, y_val)
v_r2, v_rmse = score(y_val,  best_rf.predict(X_val))
t_r2, t_rmse = score(y_test, best_rf.predict(X_test))
results.append({"model": "Random Forest", "val_r2": v_r2, "val_rmse": v_rmse,
                "test_r2": t_r2, "test_rmse": t_rmse, "best_params": bp})
test_preds_all["Random Forest"] = best_rf.predict(X_test)
print(f"  val R2={v_r2:.4f}  test R2={t_r2:.4f}  params={bp}")

# 5. Gradient Boosting
print("Fitting Gradient Boosting (grid search)...")
gb_grid = {
    "n_estimators":  [200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth":     [3, 5],
    "random_state":  [42]
}
best_gb, bp, _ = grid_search(GradientBoostingRegressor, gb_grid, X_train, y_train, X_val, y_val)
v_r2, v_rmse = score(y_val,  best_gb.predict(X_val))
t_r2, t_rmse = score(y_test, best_gb.predict(X_test))
results.append({"model": "Gradient Boosting", "val_r2": v_r2, "val_rmse": v_rmse,
                "test_r2": t_r2, "test_rmse": t_rmse, "best_params": bp})
test_preds_all["Gradient Boosting"] = best_gb.predict(X_test)
print(f"  val R2={v_r2:.4f}  test R2={t_r2:.4f}  params={bp}")

# 6. SVR — subsample train to 2000 rows for speed (SVR is O(n^2))
print("Fitting SVR (grid search, subsampled train)...")
rng = np.random.default_rng(42)
sub_idx = rng.choice(len(Xs_tr), size=min(2000, len(Xs_tr)), replace=False)
Xs_tr_sub = Xs_tr[sub_idx]
y_tr_sub  = y_train[sub_idx]
svr_grid = {
    "C":       [1.0, 10.0, 100.0],
    "epsilon": [0.1, 0.5],
    "kernel":  ["rbf"]
}
best_svr, bp, _ = grid_search(SVR, svr_grid, Xs_tr_sub, y_tr_sub, Xs_val, y_val)
# refit on full train with best params
best_svr_full = SVR(**{k: v for k, v in bp.items()})
best_svr_full.fit(Xs_tr, y_train)
v_r2, v_rmse = score(y_val,  best_svr_full.predict(Xs_val))
t_r2, t_rmse = score(y_test, best_svr_full.predict(Xs_test))
results.append({"model": "SVR", "val_r2": v_r2, "val_rmse": v_rmse,
                "test_r2": t_r2, "test_rmse": t_rmse, "best_params": bp})
test_preds_all["SVR"] = best_svr_full.predict(Xs_test)
print(f"  val R2={v_r2:.4f}  test R2={t_r2:.4f}  params={bp}")

# ── Save results ───────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df = results_df[["model", "val_r2", "val_rmse", "test_r2", "test_rmse", "best_params"]]
results_df.to_csv(os.path.join(HERE, "baseline_results.csv"), index=False)
test_preds_all.to_csv(os.path.join(HERE, "baseline_predictions.csv"), index=False)

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(results_df[["model", "val_r2", "test_r2", "test_rmse"]].to_string(index=False))
print("\nSaved: baseline_results.csv, baseline_predictions.csv")
