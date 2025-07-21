import itertools
import math
import os
import sys
import time
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.model_selection as sk_model
import sklearn.preprocessing as sk_prep
import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from dataloaders.load_ecotox import load_ecotox_data
from ecotox_datasets.FM_dataset import EcotoxFMDataset
from models.variational_fm import VariationalFactorizationMachine, train_variational_fm
from models.GibbsBFM import GibbsBFM 

def rmse_on_loader(model: VariationalFactorizationMachine, loader: DataLoader, device: torch.device) -> float:
    # This function remains for the VFM
    model.eval()
    mse_sum, n_obs = 0.0, 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            if Xb.is_sparse:
                Xb = Xb.to_dense()
            preds = model(Xb, sample=False)
            mse_sum += ((preds - yb) ** 2).sum().item()
            n_obs += yb.numel()
    return math.sqrt(mse_sum / n_obs)

def rmse_on_data(model: GibbsBFM, X: sp.csr_matrix, y: np.ndarray) -> float:
    """Calculates RMSE for a model that predicts on a full dataset."""
    preds = model.predict(X)
    mse = np.mean((preds - y) ** 2)
    return math.sqrt(mse)


def main_gibbs() -> None:
    print("\n" + "="*80)
    print("RUNNING EXPERIMENT FOR GIBBS SAMPLING BFM")
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
        "species": sk_prep.OneHotEncoder(),
        "CAS": sk_prep.OneHotEncoder(),
        "duration": sk_prep.OneHotEncoder(),
        "tax_family": sk_prep.OneHotEncoder(),
        "tax_class": sk_prep.OneHotEncoder(),
    }

    Xi = enc_dict["species"].fit_transform(full_data[["species"]])
    Xj = enc_dict["CAS"].fit_transform(full_data[["CAS"]])
    Xd = enc_dict["duration"].fit_transform(full_data[["duration"]])
    Xt = enc_dict["tax_family"].fit_transform(full_data[["tax_family"]])
    Xe = enc_dict["tax_class"].fit_transform(full_data[["tax_class"]])

    X_cat = sp.hstack([Xi, Xj, Xd, Xt, Xe], format="csr")
    y_centered_np = y_centered

    kfold = sk_model.KFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(kfold.split(X_cat, y_centered_np))

    GIBBS_CFG = dict(k=16, n_iter=200, n_burnin=100)

    print(f"\n>>> Gibbs config: {GIBBS_CFG}")
    tic = time.time()
    rmses: List[float] = []

    for fold, (tr_idx, va_idx) in enumerate(cv_splits, 1):
        X_tr = X_cat[tr_idx]
        X_va = X_cat[va_idx]
        y_tr = y_centered_np[tr_idx]
        y_va = y_centered_np[va_idx]

        model = GibbsBFM(
            n_features=X_tr.shape[1],
            k=GIBBS_CFG["k"],
        )

        model.fit(X_tr, y_tr, n_iter=GIBBS_CFG["n_iter"], n_burnin=GIBBS_CFG["n_burnin"])

        rmse_val = rmse_on_data(model, X_va, y_va)
        rmses.append(rmse_val)
        print(f"   Fold {fold}: RMSE={rmse_val:.4f}")

    mean_rmse, std_rmse = float(np.mean(rmses)), float(np.std(rmses))
    print(
        f"→ mean ± std RMSE: {mean_rmse:.4f} ± {std_rmse:.4f} | elapsed {time.time() - tic:.1f}s"
    )

if __name__ == "__main__":
    main_gibbs()