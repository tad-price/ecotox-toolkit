import numpy as np
import pandas as pd
import pickle, pathlib
import time
import scipy.sparse as sp
import sklearn.model_selection as sk_model
import sklearn.preprocessing as sk_prep
import sklearn.impute as sk_impute
import os
from os import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from dataloaders.load_ecotox import load_ecotox_data
from models.BFM_rendle import BayesianFactorizationMachine


def rmse_on_data(model, X: sp.csr_matrix, y: np.ndarray) -> float:
    preds = model.predict(X)
    mse = np.mean((preds - y) ** 2)
    return np.sqrt(mse)


def main_bfm_feature_sweep() -> None:
    print("\n" + "="*80)
    print("RUNNING FEATURE ADDITION SWEEP WITH UNBLOCKED GIBBS BFM")
    print("="*80 + "\n")

    DATA_DIR = "/home/tad/Desktop/Thesisfiles/ThesisCode/ecotox-toolkit/data_files"
    adore_path = os.path.join(DATA_DIR, "ecotox_mortality_processed.csv")
    chemicals_path = os.path.join(DATA_DIR, "ecotox_properties_with-oecd-function.csv")

    full_data, y_centered = load_ecotox_data(
        adore_path=adore_path,
        chemicals_path=chemicals_path,
        use_selfies=False,
        use_mol2vec=False,
        use_fingerprint=False,
        shuffle=True,
        random_state=42,
    )

    full_data["duration"] = pd.Categorical(full_data["duration"].astype(int))
    full_data["chem_mw"] = np.log(full_data["chem_mw"])

    enc_dict = {
        "species": sk_prep.OneHotEncoder(handle_unknown='ignore'),
        "CAS": sk_prep.OneHotEncoder(handle_unknown='ignore'),
        "duration": sk_prep.OneHotEncoder(handle_unknown='ignore'),
        "tax_family": sk_prep.OneHotEncoder(handle_unknown='ignore'),
        "tax_class": sk_prep.OneHotEncoder(handle_unknown='ignore'),
    }
    for col, enc in enc_dict.items():
        enc.fit(full_data[[col]])

    # GroupKFold on triplet (CAS, species, duration)
    triplet_id = pd.factorize(
        full_data["CAS"].astype(str) + "_" +
        full_data["species"].astype(str) + "_" +
        full_data["duration"].astype(str)
    )[0]
    gkf = sk_model.GroupKFold(n_splits=5)
    cv_splits = list(gkf.split(full_data, y_centered, groups=triplet_id))

    BFM_CFG = dict(k=32, n_iter=3, n_burn=2)
    print(f">>> BFM config: {BFM_CFG}")

    CLOGP_CANDIDATES = ["chem_rdkit_clogp", "chem_mordred_SLogP"]
    clogp_col = next((c for c in CLOGP_CANDIDATES if c in full_data.columns), None)

    base_num_cols = ["chem_mw"] + ([clogp_col] if clogp_col is not None else [])
    if clogp_col is None:
        print("! Warning: neither 'chem_rdkit_clogp' nor 'chem_mordred_SLogP' present; "
              "baseline will use only 'chem_mw'.")

    ordered_new_feats = [
        "chem_mordred_FilterItLogS",
        "chem_mordred_TopoPSA",
        "chem_mordred_apol",
        "chem_mordred_bpol",
        "chem_mordred_SMR",
        "chem_mordred_VMcGowan",
        "chem_mordred_LabuteASA",
        "chem_pcp_heavy_atom_count",
        "chem_mordred_AMW",
        "chem_mordred_nHBAcc",
        "chem_mordred_nHBDon",
        "chem_mordred_nRot",
        "chem_mordred_nAromAtom",
        "chem_mordred_nRing",
        "chem_mordred_nCl",
        "chem_mordred_nBr",
        "chem_mordred_nX",
        "chem_OH_count",
        "chem_pcp_doublebonds_count",
        "chem_mordred_ECIndex",
    ]

    # Keep only those actually present; warn otherwise
    present_new_feats = [c for c in ordered_new_feats if c in full_data.columns]
    missing = sorted(set(ordered_new_feats) - set(present_new_feats))
    if len(missing) > 0:
        print(f"! Warning: {len(missing)} requested descriptors not found and will be skipped:\n  {missing}")

    # -------------------------------------------
    # Run cumulative addition: base + first i cols
    # -------------------------------------------
    results = []
    pathlib.Path("artifacts").mkdir(exist_ok=True)

    sweep_start = time.time()
    for i in range(0, len(present_new_feats) + 1):
        add_cols = present_new_feats[:i]
        use_num_cols = base_num_cols + add_cols

        print("\n" + "-"*80)
        if i == 0:
            print(f"BASELINE (no added features) continuous cols: {use_num_cols}")
        else:
            print(f"ADDING #{i}: '{present_new_feats[i-1]}'  | Total continuous cols now: {len(use_num_cols)}")
        print("-"*80)

        fold_rmses = []
        step_tic = time.time()

        for fold, (tr_idx, va_idx) in enumerate(cv_splits, 1):
            df_tr, df_va = full_data.iloc[tr_idx], full_data.iloc[va_idx]
            y_tr, y_va = y_centered[tr_idx], y_centered[va_idx]

            # --- categorical blocks (fit once above; transform per split) ---
            Xi_tr = enc_dict["species"].transform(df_tr[["species"]])
            Xj_tr = enc_dict["CAS"].transform(df_tr[["CAS"]])
            Xd_tr = enc_dict["duration"].transform(df_tr[["duration"]])
            Xt_tr = enc_dict["tax_family"].transform(df_tr[["tax_family"]])
            Xe_tr = enc_dict["tax_class"].transform(df_tr[["tax_class"]])

            Xi_va = enc_dict["species"].transform(df_va[["species"]])
            Xj_va = enc_dict["CAS"].transform(df_va[["CAS"]])
            Xd_va = enc_dict["duration"].transform(df_va[["duration"]])
            Xt_va = enc_dict["tax_family"].transform(df_va[["tax_family"]])
            Xe_va = enc_dict["tax_class"].transform(df_va[["tax_class"]])

            # --- numeric block: median impute + standardize (fit on train only) ---
            imputer = sk_impute.SimpleImputer(strategy="median")
            scaler = sk_prep.StandardScaler(with_mean=True, with_std=True)

            num_tr = imputer.fit_transform(df_tr[use_num_cols])
            num_tr = scaler.fit_transform(num_tr)            # dense (n_tr, p)
            num_va = imputer.transform(df_va[use_num_cols])
            num_va = scaler.transform(num_va)                # dense (n_va, p)

            X_tr = sp.hstack([Xi_tr, Xj_tr, Xd_tr, Xt_tr, Xe_tr, sp.csr_matrix(num_tr)], format="csr")
            X_va = sp.hstack([Xi_va, Xj_va, Xd_va, Xt_va, Xe_va, sp.csr_matrix(num_va)], format="csr")

            # --- model ---
            model = BayesianFactorizationMachine(n_features=X_tr.shape[1], k=BFM_CFG["k"])
            model.fit(X_tr, y_tr, n_iter=BFM_CFG["n_iter"], n_burn=BFM_CFG["n_burn"])

            rmse_val = rmse_on_data(model, X_va, y_va)
            fold_rmses.append(float(rmse_val))
            print(f"   Fold {fold}: RMSE={rmse_val:.4f}")

        mean_rmse, std_rmse = float(np.mean(fold_rmses)), float(np.std(fold_rmses))
        elapsed = time.time() - step_tic

        print(f"→ Continuous cols: {len(use_num_cols)} "
              f"| mean ± std RMSE: {mean_rmse:.4f} ± {std_rmse:.4f} "
              f"| step time {elapsed:.1f}s")

        results.append({
            "n_added": i,
            "added_feature": None if i == 0 else present_new_feats[i-1],
            "n_continuous_total": len(use_num_cols),
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
            "fold_rmses": fold_rmses,
            "elapsed_s": elapsed,
        })

    total_elapsed = time.time() - sweep_start
    res_df = pd.DataFrame(results)
    res_path = "artifacts/feature_addition_results.csv"
    res_df.to_csv(res_path, index=False)
    print("\n" + "="*80)
    print(f"FEATURE ADDITION SWEEP COMPLETE | total elapsed {total_elapsed/60:.1f} min")
    print(f"Results saved to: {res_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main_bfm_feature_sweep()
