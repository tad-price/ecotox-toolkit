#!/usr/bin/env python
"""
Factorization‑Machine for Ecotox toxicity prediction

• 5‑fold **GroupKFold** — every replicate that shares the same
  (CAS, species, duration) triplet is confined to a single fold.

• Two continuous molecular descriptors are added as dense features
  and standardised *inside each fold*:

    – chem_mw  (log‑scaled molecular weight)
    – chem_rdkit_clogp
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import os, sys, time
import numpy as np
import pandas as pd
import scipy.sparse
import sklearn.preprocessing   as sk_prep
import sklearn.model_selection as sk_model
import torch
from torch.utils.data import DataLoader

# Local project code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloaders.load_ecotox         import load_ecotox_data
from ecotox_datasets.FM_dataset      import EcotoxFMDataset
from models.A0_FM_fill               import FactorizationMachine
from training.train_FM               import train_model
from evaluate_performance.rmse_FM    import evaluate_rmse


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:

    # ---------------------------- paths / device ---------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    root = "/home/tad/Desktop/Thesisfiles/ThesisCode/ecotox-toolkit/data_files"
    adore_path     = f"{root}/ecotox_mortality_processed.csv"
    chemicals_path = f"{root}/ecotox_properties_with-oecd-function.csv"

    # ---------------------------- load data --------------------------
    full_data, y_centered = load_ecotox_data(
        adore_path      = adore_path,
        chemicals_path  = chemicals_path,
        use_selfies     = False,
        use_mol2vec     = False,
        use_fingerprint = False,
        shuffle         = True,
        random_state    = 42,
    )

    # ---------------------------- feature prep -----------------------
    full_data["duration"] = pd.Categorical(full_data["duration"].astype(int))
    full_data["chem_mw"]  = np.log(full_data["chem_mw"])

    cont_cols = ["chem_mw", "chem_rdkit_clogp"]  # numerical features

    # ---------------------------- encoders ---------------------------
    enc_species    = sk_prep.OneHotEncoder()
    enc_cas        = sk_prep.OneHotEncoder()
    enc_duration   = sk_prep.OneHotEncoder()
    enc_tax_family = sk_prep.OneHotEncoder()
    enc_tax_class  = sk_prep.OneHotEncoder()

    Xi = enc_species.fit_transform(full_data[["species"]])
    Xj = enc_cas.fit_transform(full_data[["CAS"]])
    Xd = enc_duration.fit_transform(full_data[["duration"]])
    Xt = enc_tax_family.fit_transform(full_data[["tax_family"]])
    Xe = enc_tax_class.fit_transform(full_data[["tax_class"]])

    X_cats = scipy.sparse.hstack([Xi, Xj, Xd, Xt, Xe], format="csr")

    # ---------------------------- group IDs --------------------------
    triplet_id = pd.factorize(
        full_data["CAS"].astype(str) + "_"
        + full_data["species"].astype(str) + "_"
        + full_data["duration"].astype(str)
    )[0]

    gkf          = sk_model.GroupKFold(n_splits=5)
    fold_splits  = list(gkf.split(X_cats, y_centered, groups=triplet_id))

    # ---------------------------- hyper‑params -----------------------
    param_grid = dict(k=[32], lr=[0.001], weight_decay=[0.0001], epochs=[100])

    # ----------------------------------------------------------------
    # CV loop
    # ----------------------------------------------------------------
    for k_ in param_grid["k"]:
        for lr_ in param_grid["lr"]:
            for wd_ in param_grid["weight_decay"]:
                for n_epochs_ in param_grid["epochs"]:

                    print(f"\n>>> config: k={k_}, lr={lr_}, wd={wd_}, epochs={n_epochs_}")
                    rmse_scores, t0 = [], time.time()

                    for fold, (tr_idx, val_idx) in enumerate(fold_splits, 1):
                        # ---------- categorical -------------------------
                        X_cats_tr  = X_cats[tr_idx]
                        X_cats_val = X_cats[val_idx]
                        y_tr, y_val = y_centered[tr_idx], y_centered[val_idx]

                        # ---------- continuous (per‑fold scaling) -------
                        scaler       = sk_prep.StandardScaler()
                        X_cont_tr_np = scaler.fit_transform(
                            full_data.loc[tr_idx, cont_cols].astype(np.float32)
                        )
                        X_cont_val_np = scaler.transform(
                            full_data.loc[val_idx, cont_cols].astype(np.float32)
                        )
                        X_cont_tr  = scipy.sparse.csr_matrix(X_cont_tr_np)
                        X_cont_val = scipy.sparse.csr_matrix(X_cont_val_np)

                        # ---------- final design matrix ---------------
                        X_tr  = scipy.sparse.hstack([X_cats_tr,  X_cont_tr],  format="csr")
                        X_val = scipy.sparse.hstack([X_cats_val, X_cont_val], format="csr")

                        # ---------- datasets / loaders ----------------
                        train_ds = EcotoxFMDataset(X_tr,  y_tr)
                        val_ds   = EcotoxFMDataset(X_val, y_val)

                        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
                        val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

                        # ---------- model ------------------------------
                        model = FactorizationMachine(
                            n_features=X_tr.shape[1],
                            k=k_
                        ).to(device)

                        # ---------- training ---------------------------
                        train_model(
                            model        = model,
                            train_loader = train_loader,
                            val_loader   = val_loader,
                            n_epochs     = n_epochs_,
                            lr           = lr_,
                            weight_decay = wd_,
                            device       = device,
                        )

                        # ---------- evaluation -------------------------
                        rmse_val = evaluate_rmse(model, val_loader, device)
                        rmse_scores.append(rmse_val)
                        print(f"   Fold {fold}: RMSE={rmse_val:.4f}")

                    print(f"→ mean ± std RMSE: {np.mean(rmse_scores):.4f} ± "
                          f"{np.std(rmse_scores):.4f} | elapsed: {time.time() - t0:.1f}s")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
