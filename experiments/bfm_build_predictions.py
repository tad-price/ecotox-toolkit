from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.model_selection as sk_model
import sklearn.preprocessing as sk_prep
import sklearn.impute as sk_impute
from tqdm.auto import trange

ROOT_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(ROOT_DIR))

from dataloaders.load_ecotox import load_ecotox_data
from models.BFM_rendle import BayesianFactorizationMachine


def make_design_cats(df: pd.DataFrame, enc_dict: Dict[str, sk_prep.OneHotEncoder]) -> sp.csr_matrix:
    """Creates the one-hot encoded design matrix (categorical block only)."""
    Xi = enc_dict["species"].transform(df[["species"]])
    Xj = enc_dict["CAS"].transform(df[["CAS"]])
    Xd = enc_dict["duration"].transform(df[["duration"]])
    Xt = enc_dict["tax_family"].transform(df[["tax_family"]])
    Xe = enc_dict["tax_class"].transform(df[["tax_class"]])
    return sp.hstack([Xi, Xj, Xd, Xt, Xe], format="csr")


def main() -> None:
    print("\n" + "=" * 80)
    print("5-FOLD BFM WITH SEPARATE OOF UNCERTAINTY SAVING")
    print("=" * 80 + "\n")

    # ----------------------------- data -----------------------------------
    DATA_DIR = Path("/home/tad/Desktop/Thesisfiles/ThesisCode/ecotox-toolkit/data_files")
    adore_path     = DATA_DIR / "ecotox_mortality_processed.csv"
    chemicals_path = DATA_DIR / "ecotox_properties_with-oecd-function.csv"

    full_data, y_centered = load_ecotox_data(
        adore_path      = adore_path,
        chemicals_path  = chemicals_path,
        use_selfies     = False,
        use_mol2vec     = False,
        use_fingerprint = False,
        shuffle         = True,
        random_state    = 42,
    )

    full_data["duration"] = pd.Categorical(full_data["duration"].astype(int))
    full_data["chem_mw"]  = np.log(full_data["chem_mw"])
    num_cols = ["chem_mw", "chem_rdkit_clogp"]

    enc_dict: Dict[str, sk_prep.OneHotEncoder] = {
        "species":     sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "CAS":         sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "duration":    sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "tax_family":  sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "tax_class":   sk_prep.OneHotEncoder(handle_unknown="ignore"),
    }
    for col, enc in enc_dict.items():
        enc.fit(full_data[[col]])

    # -------------------- group-based 5-fold split ------------------------
    # groups are unique (CAS, species, duration) triplets
    triplet_id = pd.factorize(
        full_data["CAS"].astype(str) + "_" +
        full_data["species"].astype(str) + "_" +
        full_data["duration"].astype(str)
    )[0]

    gkf    = sk_model.GroupKFold(n_splits=5)
    splits = list(gkf.split(full_data, y_centered, groups=triplet_id))

    # persist splits so that downstream runs (e.g. stacking) reuse them
    ARTIFACTS = Path("artifacts_final"); ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(splits, ARTIFACTS / "cv_splits.pkl")

    # -------------------- allocate OOF containers -------------------------
    N = len(full_data)
    oof_mean            = np.empty(N, dtype=np.float32)
    oof_epistemic_var   = np.empty(N, dtype=np.float32)
    oof_aleatoric_var   = np.empty(N, dtype=np.float32)

    BFM_CFG = dict(k=32, n_iter=800, n_burn=600)
    print(f"BFM config: {BFM_CFG}\n")

    tic = time.time()

    # -------------------- cross-validation loop ---------------------------
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        df_tr, df_va = full_data.iloc[tr_idx], full_data.iloc[va_idx]
        y_tr, y_va   = y_centered[tr_idx], y_centered[va_idx]

        # categorical design (same logic as before)
        X_tr_cat = make_design_cats(df_tr, enc_dict)
        X_va_cat = make_design_cats(df_va, enc_dict)

        # numeric block: median impute + standardize (fit on train, transform val)
        if num_cols:
            imputer = sk_impute.SimpleImputer(strategy="median")
            scaler  = sk_prep.StandardScaler(with_mean=True, with_std=True)

            num_tr = imputer.fit_transform(df_tr[num_cols])
            num_tr = scaler.fit_transform(num_tr)
            num_va = imputer.transform(df_va[num_cols])
            num_va = scaler.transform(num_va)

            X_tr_num = sp.csr_matrix(num_tr)
            X_va_num = sp.csr_matrix(num_va)

            X_tr = sp.hstack([X_tr_cat, X_tr_num], format="csr")
            X_va = sp.hstack([X_va_cat, X_va_num], format="csr")
        else:
            X_tr, X_va = X_tr_cat, X_va_cat

        model = BayesianFactorizationMachine(
            n_features=X_tr.shape[1], k=BFM_CFG["k"]
        )
        model.fit(
            X_tr, y_tr,
            n_iter = BFM_CFG["n_iter"],
            n_burn = BFM_CFG["n_burn"],
        )

        # ---------- OOF predictions + uncertainty decomposition -----------
        X2_va   = X_va.copy();  X2_va.data **= 2
        n_draws = len(model.samples)
        draws   = np.empty((n_draws, len(va_idx)), dtype=np.float64)

        for d, s in enumerate(model.samples):
            w0, w, v = s["w0"], s["w"], s["v"]
            q        = X_va @ v
            inter    = 0.5 * ((q ** 2) - X2_va @ (v ** 2)).sum(axis=1)
            draws[d] = w0 + X_va @ w + inter

        oof_mean[va_idx]          = draws.mean(axis=0, dtype=np.float64)
        oof_epistemic_var[va_idx] = draws.var(axis=0,  dtype=np.float64, ddof=0)
        oof_aleatoric_var[va_idx] = np.mean(1.0 / np.array([s["alpha"] for s in model.samples]))

        rmse = np.sqrt(np.mean((oof_mean[va_idx] - y_va) ** 2))
        print(f"  Fold {fold}: RMSE = {rmse:.4f}  |  {n_draws} draws")

    # -------------------- save artefacts ----------------------------------
    np.save(ARTIFACTS / "oof_mean_64.npy",           oof_mean.astype(np.float32))
    np.save(ARTIFACTS / "oof_epistemic_var_64.npy",  oof_epistemic_var.astype(np.float32))
    np.save(ARTIFACTS / "oof_aleatoric_var_64.npy",  oof_aleatoric_var.astype(np.float32))
    joblib.dump(enc_dict, ARTIFACTS / "encoders_64.pkl")

    print("\n→ finished in %.1fs. All uncertainty files saved to '%s'"
          % (time.time() - tic, ARTIFACTS))


if __name__ == "__main__":
    main()
