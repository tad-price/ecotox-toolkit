from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.impute as sk_impute
import sklearn.model_selection as sk_model
import sklearn.preprocessing as sk_prep
from tqdm.auto import tqdm, trange

# --- Plotting Dependency ---
import matplotlib.pyplot as plt

# --- Ecotox Data Loader ---
try:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    sys.path.append(str(ROOT_DIR))
    from dataloaders.load_ecotox import load_ecotox_data
except ImportError:
    print("Error: Could not import 'load_ecotox_data'.")
    print("Please ensure the script is run from a location where 'dataloaders/load_ecotox.py' is accessible.")
    sys.exit(1)


# ==============================================================================
# BFM MODEL (CORRECTED AND STABILIZED)
# ==============================================================================
class BayesianFactorizationMachine:
    """
    Efficient O(k·Nₙz) Gibbs sampler for 2-way Bayesian Factorization Machines
    after Freudenthaler, Schmidt-Thieme & Rendle (2011).
    """
    def __init__(self, n_features: int, k: int):
        self.n_features = n_features
        self.k = k
        self.samples = []
        self.alpha_a0 = self.alpha_b0 = 1.0
        self.gamma_0 = 1.0
        self.mu_0 = 0.0
        self.alpha_l = self.beta_l = 1.0

    def fit(self, X: sp.csr_matrix, y: np.ndarray,
            n_iter: int = 150, n_burn: int = 50, random_state: int | None = None,
            verbose: bool = True):

        rng = np.random.default_rng(random_state)
        n_obs, p = X.shape
        assert p == self.n_features

        # parameters
        w0 = 0.0
        w  = np.zeros(p)
        v  = rng.normal(0.0, 0.1, size=(p, self.k))

        # hyper-parameters (scalar)
        alpha     = 1.0
        mu_w      = 0.0; lam_w  = 1.0
        mu_v      = 0.0; lam_v  = 1.0

        # helpers
        X2 = X.copy(); X2.data **= 2
        q  = X @ v
        interact = 0.5 * ((q ** 2) - X2 @ (v ** 2)).sum(axis=1)
        y_hat    = w0 + X @ w + interact
        e        = y - y_hat

        iterator = trange(n_iter, desc="BFM-Gibbs", leave=False) if verbose else range(n_iter)
        for it in iterator:
            # ----- sample w0 ---------------------------------------------
            var_w0 = 1.0 / (self.gamma_0 + n_obs * alpha)
            mean_w0 = var_w0 * (self.gamma_0 * self.mu_0 + alpha * np.sum(e + w0))
            new_w0 = rng.normal(mean_w0, np.sqrt(var_w0))
            e += w0 - new_w0
            w0 = new_w0

            # ----- sample linear part w ----------------------------------
            var_mu_w = 1.0 / (self.gamma_0 + p * lam_w)
            mean_mu_w = var_mu_w * lam_w * w.sum()
            mu_w = rng.normal(mean_mu_w, np.sqrt(var_mu_w))
            lam_w = rng.gamma(self.alpha_l + 0.5 * p, 1.0 / (self.beta_l + 0.5 * np.square(w - mu_w).sum()))
            for j in range(p):
                col = X.getcol(j).tocsc()
                idx = col.indices
                if idx.size == 0: continue
                x = col.data
                e_j = e[idx] + x * w[j]
                var = 1.0 / (lam_w + alpha * np.square(x).sum())
                mean = var * (lam_w * mu_w + alpha * np.dot(x, e_j))
                new_wj = rng.normal(mean, np.sqrt(var))
                e[idx] += x * (w[j] - new_wj)
                w[j] = new_wj

            # ----- sample latent matrix v ----------------------------------
            var_mu_v = 1.0 / (self.gamma_0 + p * self.k * lam_v)
            mean_mu_v = var_mu_v * lam_v * v.sum()
            mu_v = rng.normal(mean_mu_v, np.sqrt(var_mu_v))
            lam_v = rng.gamma(self.alpha_l + 0.5 * p * self.k, 1.0 / (self.beta_l + 0.5 * np.square(v - mu_v).sum()))
            for j in range(p):
                col = X.getcol(j).tocsc()
                idx = col.indices
                if idx.size == 0: continue
                x = col.data
                for f in range(self.k):
                    v_old = v[j, f]
                    h = x * (q[idx, f] - x * v_old)
                    denom = max(lam_v + alpha * np.dot(h, h), 1e-12)
                    var = 1.0 / denom
                    mean = var * (lam_v * mu_v + alpha * np.dot(h, e[idx] + h * v_old))
                    v_new = rng.normal(mean, np.sqrt(var))
                    delta = v_old - v_new
                    e[idx] += delta * h
                    q[idx, f] -= x * delta
                    v[j, f] = v_new

            # ----- sample global precision α ---------------------------
            alpha = rng.gamma(self.alpha_a0 + 0.5 * n_obs, 1.0 / (self.alpha_b0 + 0.5 * np.dot(e, e)))
            if it >= n_burn:
                self.samples.append(dict(w0=w0, w=w.copy(), v=v.copy(), alpha=alpha))

    def predict(self, X: sp.csr_matrix) -> np.ndarray: return self.predict_with_uncertainty(X)[0]

    def predict_with_uncertainty(self, X: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
        if not self.samples: raise RuntimeError("Call `fit` first.")
        X2 = X.copy(); X2.data **= 2
        draws = np.empty((len(self.samples), X.shape[0]), dtype=np.float64)
        for d, s in enumerate(self.samples):
            w0, w, v = s["w0"], s["w"], s["v"]
            q = X @ v; inter = 0.5 * ((q ** 2) - X2 @ (v ** 2)).sum(axis=1)
            draws[d] = w0 + X @ w + inter
        return draws.mean(axis=0), draws.var(axis=0, ddof=0)

# ==============================================================================
# DATA HELPERS (UNCHANGED)
# ==============================================================================
def make_design_cats(df: pd.DataFrame, enc_dict: Dict[str, sk_prep.OneHotEncoder]) -> sp.csr_matrix:
    Xi, Xj, Xd, Xt, Xe = (enc_dict[c].transform(df[[c]]) for c in ["species", "CAS", "duration", "tax_family", "tax_class"])
    return sp.hstack([Xi, Xj, Xd, Xt, Xe], format="csr")

def featurize(df: pd.DataFrame, enc_dict: Dict[str, sk_prep.OneHotEncoder], num_cols: list[str], imputer: sk_impute.SimpleImputer | None = None, scaler: sk_prep.StandardScaler | None = None) -> Tuple[sp.csr_matrix, sk_impute.SimpleImputer, sk_prep.StandardScaler]:
    X_cat = make_design_cats(df, enc_dict)
    if not num_cols: return X_cat, None, None
    if imputer is None: imputer = sk_impute.SimpleImputer(strategy="median"); num_transformed = imputer.fit_transform(df[num_cols])
    else: num_transformed = imputer.transform(df[num_cols])
    if scaler is None: scaler = sk_prep.StandardScaler(); num_transformed = scaler.fit_transform(num_transformed)
    else: num_transformed = scaler.transform(num_transformed)
    return sp.hstack([X_cat, sp.csr_matrix(num_transformed)], format="csr"), imputer, scaler


# ==============================================================================
# ACTIVE LEARNING SIMULATION (UNCHANGED)
# ==============================================================================
def active_learning_simulation():
    print("\n" + "=" * 80)
    print("ACTIVE LEARNING SIMULATION: COMPARING SAMPLING STRATEGIES")
    print("=" * 80 + "\n")

    DATA_DIR = Path("/home/tad/Desktop/Thesisfiles/ThesisCode/ecotox-toolkit/data_files")
    adore_path, chemicals_path = DATA_DIR / "ecotox_mortality_processed.csv", DATA_DIR / "ecotox_properties_with-oecd-function.csv"
    print(f"Loading data from: {DATA_DIR}")
    full_data, y_centered_np = load_ecotox_data(adore_path=adore_path, chemicals_path=chemicals_path, shuffle=True, random_state=42)
    y_centered = pd.Series(y_centered_np, index=full_data.index)
    
    full_data["duration"] = pd.Categorical(full_data["duration"].astype(int))
    full_data["chem_mw"]  = np.log(full_data["chem_mw"])
    num_cols = ["chem_mw", "chem_rdkit_clogp"]
    cat_cols = ["species", "CAS", "duration", "tax_family", "tax_class"]
    enc_dict = {col: sk_prep.OneHotEncoder(handle_unknown="ignore").fit(full_data[[col]]) for col in cat_cols}

    triplet_id = pd.factorize(full_data["CAS"].astype(str) + "_" + full_data["species"].astype(str))[0]
    indices = np.arange(len(full_data))

    train_pool_idx, test_idx = next(sk_model.GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=42).split(indices, groups=triplet_id))
    initial_train_idx, pool_idx = next(sk_model.GroupShuffleSplit(train_size=0.02, n_splits=1, random_state=42).split(train_pool_idx, groups=triplet_id[train_pool_idx]))
    initial_train_idx, pool_idx = train_pool_idx[initial_train_idx], train_pool_idx[pool_idx]
    df_test, y_test = full_data.iloc[test_idx], y_centered.iloc[test_idx]
    
    print(f"Data partitioning complete:\n"
          f"  - Initial Training Set: {len(initial_train_idx)} samples\n"
          f"  - Unlabeled Pool:       {len(pool_idx)} samples\n"
          f"  - Held-out Test Set:    {len(test_idx)} samples\n")
    
    BFM_CFG = dict(k=32, n_iter=200, n_burn=100)
    N_STEPS = 20
    BATCH_SIZE = 64
    UNCERTAINTY_TOP_K_PERCENT = 0.2

    history = {"random": [], "uncertainty_greedy": [], "uncertainty_subsampling": []}
    strategies_to_run = ["random", "uncertainty_greedy", "uncertainty_subsampling"]

    for strategy in strategies_to_run:
        print(f"\n--- Running simulation for '{strategy}' strategy ---")
        
        df_train, y_train = full_data.iloc[initial_train_idx].copy(), y_centered.iloc[initial_train_idx].copy()
        df_pool, y_pool = full_data.iloc[pool_idx].copy(), y_centered.iloc[pool_idx].copy()

        pbar = trange(N_STEPS)
        for step in pbar:
            pbar.set_description(f"Training on {len(df_train)} samples")

            X_train, imputer, scaler = featurize(df_train, enc_dict, num_cols)
            model = BayesianFactorizationMachine(n_features=X_train.shape[1], k=BFM_CFG["k"])
            model.fit(X_train, y_train.values, n_iter=BFM_CFG["n_iter"], n_burn=BFM_CFG["n_burn"], verbose=False, random_state=step)
            X_test, _, _ = featurize(df_test, enc_dict, num_cols, imputer, scaler)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(np.mean((y_pred - y_test.values) ** 2))
            history[strategy].append((len(df_train), rmse))
            pbar.set_postfix(RMSE=f"{rmse:.4f}")

            if len(df_pool) < BATCH_SIZE: print("Pool exhausted."); break

            if strategy == "random":
                select_indices = df_pool.sample(n=BATCH_SIZE, random_state=step).index
            else:
                X_pool, _, _ = featurize(df_pool, enc_dict, num_cols, imputer, scaler)
                _, epistemic_var = model.predict_with_uncertainty(X_pool)
                uncertainty_series = pd.Series(epistemic_var, index=df_pool.index)
                if strategy == "uncertainty_greedy":
                    select_indices = uncertainty_series.nlargest(BATCH_SIZE).index
                elif strategy == "uncertainty_subsampling":
                    threshold = uncertainty_series.quantile(1.0 - UNCERTAINTY_TOP_K_PERCENT)
                    candidate_indices = uncertainty_series[uncertainty_series >= threshold].index
                    num_to_sample = min(BATCH_SIZE, len(candidate_indices))
                    select_indices = pd.Series(candidate_indices).sample(n=num_to_sample, random_state=step).values

            df_train = pd.concat([df_train, df_pool.loc[select_indices]])
            y_train = pd.concat([y_train, y_pool.loc[select_indices]])
            df_pool.drop(select_indices, inplace=True)
            y_pool.drop(select_indices, inplace=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {'random': 'gray', 'uncertainty_greedy': 'firebrick', 'uncertainty_subsampling': 'dodgerblue'}
    labels = {'random': 'Random Sampling', 'uncertainty_greedy': 'Uncertainty Sampling (Greedy)', 'uncertainty_subsampling': 'Uncertainty Subsampling (Top 10%)'}
    for strategy in strategies_to_run:
        if history[strategy]:
            x, y = zip(*history[strategy])
            ax.plot(x, y, 'o-', label=labels[strategy], color=colors[strategy], lw=2)
    ax.set_title('Active Learning Performance Comparison', fontsize=16)
    ax.set_xlabel('Number of Training Samples', fontsize=12)
    ax.set_ylabel('Test Set RMSE', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    print("\n--- Final RMSE Results ---")
    for strategy in strategies_to_run:
        if history[strategy]:
            final_rmse = history[strategy][-1][1]
            print(f"  {labels[strategy]:<35}: {final_rmse:.4f}")
    fig.tight_layout()
    plt.savefig("active_learning_comparison.png", dpi=300)
    print("\n→ Simulation finished. Plot saved to 'active_learning_comparison.png'")
    plt.show()

if __name__ == "__main__":
    active_learning_simulation()
