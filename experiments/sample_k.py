import numpy as np
import pandas as pd
import pickle, pathlib
import time
import scipy.sparse as sp
import sklearn.model_selection as sk_model
import sklearn.preprocessing as sk_prep
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(ROOT_DIR))

from dataloaders.load_ecotox import load_ecotox_data
from models.sampled_k_BFM import BayesianFactorizationMachineARD

def rmse_on_data(model: BayesianFactorizationMachineARD, X: sp.csr_matrix, y: np.ndarray) -> float:
    preds = model.predict(X)
    mse = np.mean((preds - y) ** 2)
    return np.sqrt(mse)

def main_bfm_ard_experiment() -> None:
    print("\n" + "="*80)
    print("RUNNING EXPERIMENT WITH BFM + AUTOMATIC RELEVANCE DETERMINATION (ARD)")
    print("="*80 + "\n")

    # --- Data Loading ---
    DATA_DIR = "/home/tad/Desktop/Thesisfiles/ThesisCode/ecotox-toolkit/data_files"
    adore_path = os.path.join(DATA_DIR, "ecotox_mortality_processed.csv")
    chemicals_path = os.path.join(DATA_DIR, "ecotox_properties_with-oecd-function.csv")

    full_data, y_centered = load_ecotox_data(
        adore_path=adore_path,
        chemicals_path=chemicals_path,
        use_selfies=False, use_mol2vec=False, use_fingerprint=False,
        shuffle=True, random_state=42
    )
    full_data["duration"] = pd.Categorical(full_data["duration"].astype(int))

    # --- Feature Engineering ---
    enc_dict = {
        "species": sk_prep.OneHotEncoder(handle_unknown='ignore', sparse_output=True),
        "CAS": sk_prep.OneHotEncoder(handle_unknown='ignore', sparse_output=True),
        "duration": sk_prep.OneHotEncoder(handle_unknown='ignore', sparse_output=True),
        "tax_family": sk_prep.OneHotEncoder(handle_unknown='ignore', sparse_output=True),
        "tax_class": sk_prep.OneHotEncoder(handle_unknown='ignore', sparse_output=True),
    }
    for col, enc in enc_dict.items():
        enc.fit(full_data[[col]])

    # --- Cross-Validation Setup (GroupKFold) ---
    print("Setting up 5-fold GroupKFold cross-validation...")
    triplet_id = pd.factorize(
        full_data["CAS"].astype(str) + "_" +
        full_data["species"].astype(str) + "_" +
        full_data["duration"].astype(str)
    )[0]

    gkf = sk_model.GroupKFold(n_splits=5)
    cv_splits = list(gkf.split(full_data, y_centered, groups=triplet_id))
    print(f"CV setup complete. Number of splits: {len(cv_splits)}")

    # PARAMETERS FOR BFM
    BFM_CFG = dict(k=64, n_iter=100, n_burn=50) 

    print(f"\n>>> BFM-ARD config: {BFM_CFG}")
    tic = time.time()
    rmses = []
    all_fold_lam_v = []

    for fold, (tr_idx, va_idx) in enumerate(cv_splits, 1):
        print(f"\n--- Fold {fold}/5 ---")
        df_tr, df_va = full_data.iloc[tr_idx], full_data.iloc[va_idx]
        y_tr, y_va = y_centered[tr_idx], y_centered[va_idx]

        X_tr_list = [enc.transform(df_tr[[col]]) for col, enc in enc_dict.items()]
        X_tr = sp.hstack(X_tr_list, format="csr")

        X_va_list = [enc.transform(df_va[[col]]) for col, enc in enc_dict.items()]
        X_va = sp.hstack(X_va_list, format="csr")

        model = BayesianFactorizationMachineARD(
            n_features=X_tr.shape[1], k=BFM_CFG["k"]
        )
        model.fit(X_tr, y_tr, n_iter=BFM_CFG["n_iter"], n_burn=BFM_CFG["n_burn"], random_state=42+fold)

        rmse_val = rmse_on_data(model, X_va, y_va)
        rmses.append(rmse_val)
        print(f"   Fold {fold}: Validation RMSE = {rmse_val:.4f}")

        # Evaluate and report learned dimensionality
        if model.samples:
            lam_v_samples = np.array([s['lam_v'] for s in model.samples])
            mean_lam_v = lam_v_samples.mean(axis=0)
            all_fold_lam_v.append(mean_lam_v)
            
            variances = 1.0 / mean_lam_v
            variances_sorted = np.sort(variances)[::-1]
            
            threshold = 0.01 * variances_sorted[0]
            effective_k = np.sum(variances > threshold)

            print(f"   Learned Factor Relevance (Variances):")
            print(f"   - Top 5 most relevant factors (variances): {variances_sorted[:5].round(4)}")
            print(f"   - Effective K (variance > 1% of max): {effective_k}/{BFM_CFG['k']}")
            
        pathlib.Path("artifacts").mkdir(exist_ok=True)
        with open(f"artifacts/bfm_ard_fold_{fold}.pkl", "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    # --- Final Results ---
    print("\n" + "="*80 + "\nCross-Validation Summary\n" + "="*80)
    
    mean_rmse, std_rmse = float(np.mean(rmses)), float(np.std(rmses))
    print(f"→ Mean ± Std Dev RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

    if all_fold_lam_v:
        avg_variances = 1.0 / np.mean(all_fold_lam_v, axis=0)
        avg_variances_sorted = np.sort(avg_variances)[::-1]
        
        threshold = 0.01 * avg_variances_sorted[0]
        final_effective_k = np.sum(avg_variances > threshold)

        print(f"→ Average Learned Factor Variances (Top 5): {avg_variances_sorted[:63].round(4)}")
        print(f"→ Final Average Effective K: {final_effective_k}/{BFM_CFG['k']}")

    print(f"→ Total elapsed time: {time.time() - tic:.1f}s")

if __name__ == "__main__":
    main_bfm_ard_experiment()