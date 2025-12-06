import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from dataloaders.load_ecotox import load_ecotox_data

def main():
    print("Running Hierarchical BFM Analysis...")
    
    # 1. Load Data
    DATA_DIR = ROOT_DIR / "data_files"
    full_data, _ = load_ecotox_data(
        adore_path=DATA_DIR / "ecotox_mortality_processed.csv",
        chemicals_path=DATA_DIR / "ecotox_properties_with-oecd-function.csv",
        use_selfies=False, use_mol2vec=False, use_fingerprint=False,
        shuffle=True, random_state=42
    )
    
    # 2. Load Hierarchical Results
    ART_HIER = ROOT_DIR / "artifacts_hierarchical"
    oof_aleatoric = np.load(ART_HIER / "oof_aleatoric.npy")
    oof_epistemic = np.load(ART_HIER / "oof_epistemic.npy")
    
    # 3. Load Global Results (if available, for comparison)
    ART_GLOBAL = ROOT_DIR / "artifacts_final"
    try:
        oof_aleatoric_global = np.load(ART_GLOBAL / "oof_aleatoric_var_64.npy")
        has_global = True
        print("Loaded global results for comparison.")
    except FileNotFoundError:
        has_global = False
        print("Global results not found. Skipping comparison.")

    # 4. Prepare DataFrame
    df = full_data.copy()
    df["aleatoric_sd"] = np.sqrt(oof_aleatoric)
    df["epistemic_sd"] = np.sqrt(oof_epistemic)
    
    if has_global:
        df["aleatoric_sd_global"] = np.sqrt(oof_aleatoric_global)

    # 5. Group by Chemical
    chem_stats = df.groupby("CAS").agg(
        n_obs=("CAS", "size"),
        mean_aleatoric_sd=("aleatoric_sd", "mean"),
        mean_epistemic_sd=("epistemic_sd", "mean")
    )
    
    if has_global:
        chem_stats["mean_aleatoric_sd_global"] = df.groupby("CAS")["aleatoric_sd_global"].mean()

    chem_stats = chem_stats.reset_index()

    # 6. Plot: Aleatoric vs N_obs (Shrinkage Effect)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=chem_stats, x="n_obs", y="mean_aleatoric_sd", alpha=0.6, label="Hierarchical (Per-Chemical)")
    
    if has_global:
        # Plot global average as a horizontal line or scatter
        global_mean = df["aleatoric_sd_global"].mean()
        plt.axhline(global_mean, color="r", linestyle="--", label=f"Global Average (SD={global_mean:.2f})")
        
    plt.xscale("log")
    plt.xlabel("Number of Observations (log)")
    plt.ylabel("Aleatoric Standard Deviation")
    plt.title("Hierarchical BFM: Aleatoric Uncertainty vs Data Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(ART_HIER / "analysis_aleatoric_shrinkage.png")
    print(f"Saved analysis_aleatoric_shrinkage.png")
    
    # 7. Plot: Epistemic vs Aleatoric Correlation
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=chem_stats, x="mean_aleatoric_sd", y="mean_epistemic_sd", size="n_obs", sizes=(20, 200), alpha=0.6)
    plt.xlabel("Aleatoric SD")
    plt.ylabel("Epistemic SD")
    plt.title("Epistemic vs Aleatoric Uncertainty (by Chemical)")
    plt.grid(True, alpha=0.3)
    plt.savefig(ART_HIER / "analysis_epistemic_vs_aleatoric.png")
    print(f"Saved analysis_epistemic_vs_aleatoric.png")

    # 8. Print Summary Statistics
    print("\n" + "="*50)
    print("UNCERTAINTY ANALYSIS SUMMARY")
    print("="*50)
    print(f"Global Mean Aleatoric SD: {df['aleatoric_sd'].mean():.4f}")
    print(f"Min Aleatoric SD:         {df['aleatoric_sd'].min():.4f}")
    print(f"Max Aleatoric SD:         {df['aleatoric_sd'].max():.4f}")
    print("-" * 50)
    print("Top 5 'Noisiest' Chemicals (High Aleatoric SD):")
    print(chem_stats.nlargest(5, "mean_aleatoric_sd")[["CAS", "n_obs", "mean_aleatoric_sd"]])
    print("-" * 50)
    print("Top 5 'Cleanest' Chemicals (Low Aleatoric SD):")
    print(chem_stats.nsmallest(5, "mean_aleatoric_sd")[["CAS", "n_obs", "mean_aleatoric_sd"]])
    print("="*50)

if __name__ == "__main__":
    main()
