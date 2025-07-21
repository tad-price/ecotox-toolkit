# Unblocked gibbs sampling here

import numpy as np
import pandas as pd
import pickle, pathlib
import time
import scipy.sparse as sp
import sklearn.model_selection as sk_model
import sklearn.preprocessing as sk_prep
import os
from os import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from dataloaders.load_ecotox import load_ecotox_data
# Import the new model
from models.BFM_rendle import BayesianFactorizationMachine

def rmse_on_data(model, X: sp.csr_matrix, y: np.ndarray) -> float:
    preds = model.predict(X)
    mse = np.mean((preds - y) ** 2)
    return np.sqrt(mse)


def main_bfm_paper() -> None:
    print("\n" + "="*80)
    print("RUNNING EXPERIMENT WITH UNBLOCKED GIBBS BFM")
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

    kfold = sk_model.KFold(n_splits=3, shuffle=True, random_state=42)
    cv_splits = list(kfold.split(full_data))

# PARAMETERS FOR BFM
    BFM_CFG = dict(k=32, n_iter=3, n_burn=2)


    print(f"\n>>> BFM config: {BFM_CFG}")
    tic = time.time()
    rmses = []

    for fold, (tr_idx, va_idx) in enumerate(cv_splits, 1):
        df_tr, df_va = full_data.iloc[tr_idx], full_data.iloc[va_idx]
        y_tr, y_va = y_centered[tr_idx], y_centered[va_idx]

        # Note: We transform based on the encoder fitted on the full data - this could result in data leakage
        Xi_tr = enc_dict["species"].transform(df_tr[["species"]])
        Xj_tr = enc_dict["CAS"].transform(df_tr[["CAS"]])
        Xd_tr = enc_dict["duration"].transform(df_tr[["duration"]])
        Xt_tr = enc_dict["tax_family"].transform(df_tr[["tax_family"]])
        Xe_tr = enc_dict["tax_class"].transform(df_tr[["tax_class"]])
        X_tr = sp.hstack([Xi_tr, Xj_tr, Xd_tr, Xt_tr, Xe_tr], format="csr")

        Xi_va = enc_dict["species"].transform(df_va[["species"]])
        Xj_va = enc_dict["CAS"].transform(df_va[["CAS"]])
        Xd_va = enc_dict["duration"].transform(df_va[["duration"]])
        Xt_va = enc_dict["tax_family"].transform(df_va[["tax_family"]])
        Xe_va = enc_dict["tax_class"].transform(df_va[["tax_class"]])
        X_va = sp.hstack([Xi_va, Xj_va, Xd_va, Xt_va, Xe_va], format="csr")

        model = BayesianFactorizationMachine(
            n_features=X_tr.shape[1],
            k=BFM_CFG["k"],
        )

        model.fit(X_tr, y_tr, n_iter=BFM_CFG["n_iter"], n_burn=BFM_CFG["n_burn"])

        rmse_val = rmse_on_data(model, X_va, y_va)
        rmses.append(rmse_val)
        print(f"   Fold {fold}: RMSE={rmse_val:.4f}")
        pathlib.Path("artifacts").mkdir(exist_ok=True)

        with open("artifacts/bfm.pkl", "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


    mean_rmse, std_rmse = float(np.mean(rmses)), float(np.std(rmses))
    print(
        f"→ mean ± std RMSE: {mean_rmse:.4f} ± {std_rmse:.4f} | elapsed {time.time() - tic:.1f}s"
    )

if __name__ == "__main__":
    main_bfm_paper()