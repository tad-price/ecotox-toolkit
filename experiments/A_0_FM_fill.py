import os, sys, time
import numpy as np
import pandas as pd
import scipy.sparse
import sklearn.preprocessing as sk_prep
import sklearn.model_selection as sk_model
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloaders.load_ecotox import load_ecotox_data          
from ecotox_datasets.FM_dataset import EcotoxFMDataset
from models.A0_FM_fill import FactorizationMachine
from training.train_FM import train_model
from evaluate_performance.rmse_FM import evaluate_rmse

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(os.getcwd())
    adore_path = '/home/tad/Desktop/Thesisfiles/ThesisCode/ecotox-toolkit/data_files/ecotox_mortality_processed.csv'
    chemicals_path = '/home/tad/Desktop/Thesisfiles/ThesisCode/ecotox-toolkit/data_files/ecotox_properties_with-oecd-function.csv'

    full_data, y_centered = load_ecotox_data(
        adore_path=adore_path,
        chemicals_path = chemicals_path,
        use_selfies=False,
        use_mol2vec=False,
        use_fingerprint=False,
        shuffle=True,
        random_state=42,
    )

    full_data["duration"] = pd.Categorical(full_data["duration"].astype(int))

    full_data["chem_mw"] = np.log(full_data["chem_mw"])
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

    cats_with_id = scipy.sparse.hstack([Xi, Xj, Xd, Xt, Xe], format="csr")

    kfold = sk_model.KFold(n_splits=5)
    fold_splits = list(kfold.split(cats_with_id, y_centered))

    param_grid = dict(k=[32], lr=[1e-3], weight_decay=[1e-4], epochs=[100])

    for k_ in param_grid["k"]:
        for lr_ in param_grid["lr"]:
            for wd_ in param_grid["weight_decay"]:
                for n_epochs_ in param_grid["epochs"]:

                    start = time.time()
                    rmse_scores = []

                    print(f"\n>>> config: k={k_}, lr={lr_}, wd={wd_}, epochs={n_epochs_}")

                    for fold, (tr_idx, val_idx) in enumerate(fold_splits, 1):

                        X_train = cats_with_id[tr_idx]
                        X_val   = cats_with_id[val_idx]
                        y_train = y_centered[tr_idx]
                        y_val   = y_centered[val_idx]

                        train_ds = EcotoxFMDataset(X_train, y_train)
                        val_ds   = EcotoxFMDataset(X_val,   y_val)

                        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
                        val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

                        # Model
                        model = FactorizationMachine(
                            n_features=X_train.shape[1],
                            k=k_
                        ).to(device)

                        # Train
                        train_model(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            n_epochs=n_epochs_,
                            lr=lr_,
                            weight_decay=wd_,
                            device=device
                        )

                        # Evaluate
                        rmse_val = evaluate_rmse(model, val_loader, device)
                        rmse_scores.append(rmse_val)
                        print(f"   Fold {fold}: RMSE={rmse_val:.4f}")

                    print(f"→ mean ± std RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f} "
                          f"| elapsed: {time.time() - start:.1f}s")

if __name__ == "__main__":
    main()
