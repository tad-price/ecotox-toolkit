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
from tqdm.auto import trange

ROOT_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(ROOT_DIR))

from dataloaders.load_ecotox import load_ecotox_data  # noqa
from models.BFM_rendle import BayesianFactorizationMachine  # noqa


def make_design(df: pd.DataFrame, enc_dict: Dict[str, sk_prep.OneHotEncoder]) -> sp.csr_matrix:
    Xi = enc_dict["species"].transform(df[["species"]])
    Xj = enc_dict["CAS"].transform(df[["CAS"]])
    Xd = enc_dict["duration"].transform(df[["duration"]])
    Xt = enc_dict["tax_family"].transform(df[["tax_family"]])
    Xe = enc_dict["tax_class"].transform(df[["tax_class"]])
    return sp.hstack([Xi, Xj, Xd, Xt, Xe], format="csr")

# ---------------------------------------------------------------------------
# Main experiment                                                             
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 80)
    print("5‑FOLD BFM WITH OOF DISTRIBUTION SAVING")
    print("=" * 80 + "\n")

    # ----------------------------- data -----------------------------------
    DATA_DIR = Path("/home/tad/Desktop/Thesisfiles/ThesisCode/ecotox-toolkit/data_files")
    adore_path = DATA_DIR / "ecotox_mortality_processed.csv"
    chemicals_path = DATA_DIR / "ecotox_properties_with-oecd-function.csv"

    full_data, y_centered = load_ecotox_data(
        adore_path=adore_path,
        chemicals_path=chemicals_path,
        use_selfies=False,
        use_mol2vec=False,
        use_fingerprint=False,
        shuffle=True,
        random_state=42,
    )

    # simple feature transformations ---------------------------------------
    full_data["duration"] = pd.Categorical(full_data["duration"].astype(int))
    full_data["chem_mw"] = np.log(full_data["chem_mw"])

    # fit encoders once on the full data -----------------------------------
    enc_dict: Dict[str, sk_prep.OneHotEncoder] = {
        "species": sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "CAS": sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "duration": sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "tax_family": sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "tax_class": sk_prep.OneHotEncoder(handle_unknown="ignore"),
    }
    for col, enc in enc_dict.items():
        enc.fit(full_data[[col]])

    # ---------------------- cross‑validation splits -----------------------
    kfold = sk_model.KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kfold.split(full_data))

    # save the splits for reproducibility
    ARTIFACTS = Path("artifacts"); ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(splits, ARTIFACTS / "cv_splits.pkl")

    # ---------------------- containers for OOF ---------------------------
    N = len(full_data)
    oof_mean = np.empty(N, dtype=np.float32)
    oof_var  = np.empty(N, dtype=np.float32)

    # ---------------------- BFM hyper‑parameters -------------------------
    BFM_CFG = dict(k=32, n_iter=10, n_burn=5)
    print(f"BFM config: {BFM_CFG}\n")

    tic = time.time()

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        df_tr, df_va = full_data.iloc[tr_idx], full_data.iloc[va_idx]
        y_tr, y_va   = y_centered[tr_idx], y_centered[va_idx]

        # build sparse design matrices
        X_tr = make_design(df_tr, enc_dict)
        X_va = make_design(df_va, enc_dict)

        # ------------------ fit model ------------------------------------
        model = BayesianFactorizationMachine(n_features=X_tr.shape[1], k=BFM_CFG["k"])
        model.fit(X_tr, y_tr, n_iter=BFM_CFG["n_iter"], n_burn=BFM_CFG["n_burn"])

        # ------------------ predictive draws -----------------------------
        X2_va = X_va.copy(); X2_va.data **= 2
        n_draws  = len(model.samples)
        draws = np.empty((n_draws, va_idx.size), dtype=np.float64)

        for d, s in enumerate(model.samples):
            w0, w, v = s["w0"], s["w"], s["v"]
            q = X_va @ v
            inter = 0.5 * ((q ** 2) - X2_va @ (v ** 2)).sum(axis=1)
            draws[d] = w0 + X_va @ w + inter

        # ------------------ store OOF mean / variance --------------------
        oof_mean[va_idx] = draws.mean(axis=0, dtype=np.float64)
        oof_var[va_idx]  = draws.var(axis=0, dtype=np.float64, ddof=0)

        rmse = np.sqrt(np.mean((oof_mean[va_idx] - y_va) ** 2))
        print(f"  Fold {fold}: RMSE = {rmse:.4f}  |  {n_draws} draws")

    # ------------------------ save artefacts -----------------------------
    np.save(ARTIFACTS / "oof_mean.npy", oof_mean.astype(np.float32))
    np.save(ARTIFACTS / "oof_var.npy",  oof_var.astype(np.float32))
    joblib.dump(enc_dict, ARTIFACTS / "encoders.pkl")

    print("\n→ finished in %.1fs.  Files saved to '%s'" % (time.time() - tic, ARTIFACTS))


if __name__ == "__main__":
    main()
