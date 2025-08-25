# experiments/exp_bfm_ids_vs_basic.py
import os, time, pathlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.model_selection as sk_model
import sklearn.preprocessing as sk_prep
import sklearn.impute as sk_impute

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys; sys.path.append(ROOT_DIR)

from dataloaders.load_ecotox import load_ecotox_data
from models.BFM_rendle import BayesianFactorizationMachine
s

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    print("\n" + "="*80)
    print("UNBLOCKED GIBBS BFM — IDS BASELINE, THEN ADD BASIC FEATURES")
    print("="*80 + "\n")

    # ---------- paths ----------
    DATA_DIR = os.environ.get("ECOTOX_DATA_DIR", os.path.join(ROOT_DIR, "data_files"))
    adore_path = os.path.join(DATA_DIR, "ecotox_mortality_processed.csv")
    chemicals_path = os.path.join(DATA_DIR, "ecotox_properties_with-oecd-function.csv")

    # ---------- data ----------
    full_data, y_centered = load_ecotox_data(
        adore_path=adore_path,
        chemicals_path=chemicals_path,
        use_selfies=False, use_mol2vec=False, use_fingerprint=False,
        shuffle=True, random_state=42,
    )

    # canonicalize a few fields
    full_data = full_data.copy()
    full_data["duration"] = pd.Categorical(full_data["duration"].astype(int))
    # log-mw like your previous runs
    if (full_data["chem_mw"] <= 0).any():
        raise ValueError("chem_mw contains non-positive values; cannot log-transform.")
    full_data["chem_mw"] = np.log(full_data["chem_mw"])

    # choose which clogP to use
    CLOGP_CHOICES = ["chem_rdkit_clogp", "chem_mordred_SLogP"]
    clogp_col = next((c for c in CLOGP_CHOICES if c in full_data.columns), None)
    if clogp_col is None:
        print("! Warning: no clogP column found; the ‘add clogP’ step will be skipped.")

    # ---------- CV split ----------
    # Triplet holdout like your current script:
    triplet_id = pd.factorize(
        full_data["CAS"].astype(str) + "_" +
        full_data["species"].astype(str) + "_" +
        full_data["duration"].astype(str)
    )[0]
    gkf = sk_model.GroupKFold(n_splits=2)
    cv_splits = list(gkf.split(full_data, y_centered, groups=triplet_id))

    # (Optional) For chemical-holdout instead, replace groups=full_data["CAS"].astype(str)

    # ---------- config ----------
    BFM_CFG = dict(k=32, n_iter=200, n_burn=100)
    print(f">>> BFM config: {BFM_CFG}")

    steps = []
    # 0) baseline: ONLY identifiers
    steps.append(dict(
        name="ids_only",
        cat_cols=["species", "CAS", "duration"],
        num_cols=[]
    ))
    # 1) add chem_mw
    steps.append(dict(
        name="ids_plus_mw",
        cat_cols=["species", "CAS", "duration"],
        num_cols=["chem_mw"]
    ))
    # 2) add clogP (if available)
    if clogp_col is not None:
        steps.append(dict(
            name=f"ids_plus_mw_{clogp_col}",
            cat_cols=["species", "CAS", "duration"],
            num_cols=["chem_mw", clogp_col]
        ))
    # 3) add tax_class (categorical)
    steps.append(dict(
        name="ids_mw_clogp_taxclass" if clogp_col else "ids_mw_taxclass",
        cat_cols=["species", "CAS", "duration", "tax_class"],
        num_cols=["chem_mw"] + ([clogp_col] if clogp_col else [])
    ))
    # 4) add tax_family (categorical)
    steps.append(dict(
        name="ids_mw_clogp_taxclass_taxfamily" if clogp_col else "ids_mw_taxclass_taxfamily",
        cat_cols=["species", "CAS", "duration", "tax_class", "tax_family"],
        num_cols=["chem_mw"] + ([clogp_col] if clogp_col else [])
    ))

    results = []
    pathlib.Path("artifacts").mkdir(exist_ok=True)
    tic_all = time.time()

    for s_idx, step in enumerate(steps[2:], 3):
        print("\n" + "-"*80)
        print(f"STEP {s_idx}/{len(steps)}: {step['name']}")
        print(f"  Categorical: {step['cat_cols']}")
        print(f"  Numeric:     {step['num_cols']}")
        print("-"*80)

        fold_rmses = []
        step_tic = time.time()

        for fold, (tr_idx, va_idx) in enumerate(cv_splits, 1):
            df_tr = full_data.iloc[tr_idx]
            df_va = full_data.iloc[va_idx]
            y_tr, y_va = y_centered[tr_idx], y_centered[va_idx]

            # ----- categorical encoders (fit on train only) -----
            cat_blocks_tr = []
            cat_blocks_va = []
            for col in step["cat_cols"]:
                if col not in df_tr.columns:
                    raise KeyError(f"Missing categorical column: {col}")
                enc = sk_prep.OneHotEncoder(handle_unknown="ignore", sparse_output=True)
                enc.fit(df_tr[[col]])
                cat_blocks_tr.append(enc.transform(df_tr[[col]]))
                cat_blocks_va.append(enc.transform(df_va[[col]]))

            # ----- numeric impute/scale (train-only fit) -----
            num_tr = sp.csr_matrix((len(df_tr), 0))
            num_va = sp.csr_matrix((len(df_va), 0))
            if step["num_cols"]:
                for c in step["num_cols"]:
                    if c not in df_tr.columns:
                        raise KeyError(f"Missing numeric column: {c}")

                imputer = sk_impute.SimpleImputer(strategy="median")
                scaler = sk_prep.StandardScaler(with_mean=True, with_std=True)

                xtr = imputer.fit_transform(df_tr[step["num_cols"]])
                xtr = scaler.fit_transform(xtr)
                xva = imputer.transform(df_va[step["num_cols"]])
                xva = scaler.transform(xva)

                # guard against degenerate variance (rare with these cols, but safe)
                if np.any(np.isnan(xtr)) or np.any(np.isinf(xtr)):
                    raise ValueError("NaN/Inf encountered in numeric training block after scaling.")

                num_tr = sp.csr_matrix(xtr)
                num_va = sp.csr_matrix(xva)

            # ----- design matrices -----
            X_tr = sp.hstack(cat_blocks_tr + [num_tr], format="csr")
            X_va = sp.hstack(cat_blocks_va + [num_va], format="csr")

            # ----- model -----
            model = BayesianFactorizationMachine(n_features=X_tr.shape[1], k=BFM_CFG["k"])
            model.fit(X_tr, y_tr, n_iter=BFM_CFG["n_iter"], n_burn=BFM_CFG["n_burn"])

            yhat_va = model.predict(X_va)
            r = rmse(y_va, yhat_va)
            fold_rmses.append(r)
            print(f"   Fold {fold}: RMSE={r:.4f}")

        mean_rmse = float(np.mean(fold_rmses))
        std_rmse = float(np.std(fold_rmses))
        elapsed = time.time() - step_tic

        print(f"→ {step['name']} | mean ± std RMSE: {mean_rmse:.4f} ± {std_rmse:.4f} | step time {elapsed:.1f}s")

        results.append({
            "step": step["name"],
            "cat_cols": ",".join(step["cat_cols"]),
            "num_cols": ",".join(step["num_cols"]),
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
            "fold_rmses": fold_rmses,
            "elapsed_s": elapsed,
        })

    total_min = (time.time() - tic_all) / 60.0
    res_df = pd.DataFrame(results)
    out_csv = "artifacts/bfm_ids_vs_basic_results.csv"
    res_df.to_csv(out_csv, index=False)

    print("\n" + "="*80)
    print(f"COMPLETE in {total_min:.1f} min | Results -> {out_csv}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

