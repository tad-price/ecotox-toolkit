import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Add root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from models.HierarchicalBFM import HierarchicalBFM
from dataloaders.load_ecotox import load_ecotox_data
import sklearn.preprocessing as sk_prep
import sklearn.impute as sk_impute

def make_design_cats(df: pd.DataFrame, enc_dict: dict) -> sp.csr_matrix:
    Xi = enc_dict["species"].transform(df[["species"]])
    Xj = enc_dict["CAS"].transform(df[["CAS"]])
    Xd = enc_dict["duration"].transform(df[["duration"]])
    Xt = enc_dict["tax_family"].transform(df[["tax_family"]])
    Xe = enc_dict["tax_class"].transform(df[["tax_class"]])
    return sp.hstack([Xi, Xj, Xd, Xt, Xe], format="csr")

def main():
    print("Running Hierarchical BFM Experiment...")
    
    # 1. Load Data
    DATA_DIR = ROOT_DIR / "data_files"
    full_data, y_centered = load_ecotox_data(
        adore_path=DATA_DIR / "ecotox_mortality_processed.csv",
        chemicals_path=DATA_DIR / "ecotox_properties_with-oecd-function.csv",
        use_selfies=False, use_mol2vec=False, use_fingerprint=False,
        shuffle=True, random_state=42
    )
    
    full_data["duration"] = pd.Categorical(full_data["duration"].astype(int))
    full_data["chem_mw"]  = np.log(full_data["chem_mw"])
    num_cols = ["chem_mw", "chem_rdkit_clogp"]

    # 2. Prepare Encoders
    enc_dict = {
        "species":     sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "CAS":         sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "duration":    sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "tax_family":  sk_prep.OneHotEncoder(handle_unknown="ignore"),
        "tax_class":   sk_prep.OneHotEncoder(handle_unknown="ignore"),
    }
    for col, enc in enc_dict.items():
        enc.fit(full_data[[col]])

    # 3. Prepare Groups (CAS indices) for Hierarchical Model
    # We need a mapping from CAS string to integer index 0..N_groups-1
    unique_cas = full_data["CAS"].unique()
    cas_to_idx = {cas: i for i, cas in enumerate(unique_cas)}
    groups = full_data["CAS"].map(cas_to_idx).values.astype(int)
    n_groups = len(unique_cas)
    print(f"Number of groups (chemicals): {n_groups}")

    # 4. Load Splits
    ARTIFACTS = Path("artifacts_hierarchical")
    ARTIFACTS.mkdir(exist_ok=True)
    
    # Try to load existing splits to be comparable
    try:
        splits = joblib.load("artifacts_final/cv_splits.pkl")
        print("Loaded existing splits.")
    except:
        print("Could not load splits, creating new ones (not comparable!).")
        import sklearn.model_selection as sk_model
        triplet_id = pd.factorize(
            full_data["CAS"].astype(str) + "_" +
            full_data["species"].astype(str) + "_" +
            full_data["duration"].astype(str)
        )[0]
        gkf = sk_model.GroupKFold(n_splits=3)
        splits = list(gkf.split(full_data, y_centered, groups=triplet_id))

    # 5. CV Loop
    oof_mean = np.zeros(len(full_data))
    oof_epistemic = np.zeros(len(full_data))
    oof_aleatoric = np.zeros(len(full_data))
    
    # Store learned alphas for analysis (dictionary mapping CAS -> list of alphas across folds)
    cas_alpha_samples = {cas: [] for cas in unique_cas}

    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"\nFold {fold+1}/5")
        
        df_tr, df_va = full_data.iloc[tr_idx], full_data.iloc[va_idx]
        y_tr, y_va   = y_centered[tr_idx], y_centered[va_idx]
        groups_tr    = groups[tr_idx]
        
        # Prepare X
        X_tr_cat = make_design_cats(df_tr, enc_dict)
        X_va_cat = make_design_cats(df_va, enc_dict)
        
        imputer = sk_impute.SimpleImputer(strategy="median")
        scaler  = sk_prep.StandardScaler()
        
        num_tr = scaler.fit_transform(imputer.fit_transform(df_tr[num_cols]))
        num_va = scaler.transform(imputer.transform(df_va[num_cols]))
        
        X_tr = sp.hstack([X_tr_cat, sp.csr_matrix(num_tr)], format="csr")
        X_va = sp.hstack([X_va_cat, sp.csr_matrix(num_va)], format="csr")
        
        # Train Hierarchical BFM
        model = HierarchicalBFM(n_features=X_tr.shape[1], n_groups=n_groups, k=32)
        model.fit(X_tr, y_tr, groups=groups_tr, n_iter=200, n_burn=100)
        
        # Predict
        X2_va = X_va.copy(); X2_va.data **= 2
        preds = []
        
        
        groups_va = groups[va_idx]
        aleatoric_samples = []

        for s in model.samples:
            w0, w, v, alpha_vec = s["w0"], s["w"], s["v"], s["alpha_vec"]
            
            # Prediction
            q = X_va @ v
            inter = 0.5 * ((q**2) - X2_va @ (v**2)).sum(axis=1)
            pred = w0 + X_va @ w + inter
            preds.append(pred)
            
            # Aleatoric (1/alpha) for each validation point
            # alpha_vec is size n_groups. groups_va maps val rows to groups.
            alpha_vals = alpha_vec[groups_va]
            aleatoric_samples.append(1.0 / alpha_vals)
            
            # Store alphas for analysis
            for cas, idx in cas_to_idx.items():
                cas_alpha_samples[cas].append(alpha_vec[idx])

        preds = np.array(preds)
        aleatoric_samples = np.array(aleatoric_samples)
        
        oof_mean[va_idx] = preds.mean(axis=0)
        oof_epistemic[va_idx] = preds.var(axis=0)
        oof_aleatoric[va_idx] = aleatoric_samples.mean(axis=0)
        
        rmse = np.sqrt(np.mean((oof_mean[va_idx] - y_va)**2))
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")

    # 6. Save Results
    np.save(ARTIFACTS / "oof_mean.npy", oof_mean)
    np.save(ARTIFACTS / "oof_epistemic.npy", oof_epistemic)
    np.save(ARTIFACTS / "oof_aleatoric.npy", oof_aleatoric)
    
    # 7. Analysis & Plotting
    print("Generating plots...")
    df = full_data.copy()
    df["aleatoric_sd"] = np.sqrt(oof_aleatoric)
    
    # Group by CAS
    chem_stats = df.groupby("CAS").agg(
        n_obs=("CAS", "size"),
        mean_aleatoric_sd=("aleatoric_sd", "mean")
    ).reset_index()
    
    # Plot 1: Aleatoric SD vs N_obs
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=chem_stats, x="n_obs", y="mean_aleatoric_sd", alpha=0.6)
    plt.xscale("log")
    plt.xlabel("Number of Observations (log)")
    plt.ylabel("Learned Aleatoric SD")
    plt.title("Hierarchical BFM: Aleatoric Uncertainty vs Data Size")
    plt.grid(True, alpha=0.3)
    plt.savefig(ARTIFACTS / "aleatoric_vs_nobs.png")
    plt.close()
    
    # Plot 2: Distribution of Aleatoric SD
    plt.figure(figsize=(10, 6))
    sns.histplot(chem_stats["mean_aleatoric_sd"], bins=50)
    plt.xlabel("Aleatoric SD")
    plt.title("Distribution of Per-Chemical Aleatoric Uncertainty")
    plt.savefig(ARTIFACTS / "aleatoric_dist.png")
    plt.close()
    
    print(f"Done! Artifacts saved to {ARTIFACTS}")

if __name__ == "__main__":
    main()
