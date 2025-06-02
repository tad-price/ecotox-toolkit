import os
import sys
import ast
import numpy as np
import pandas as pd
import torch
import itertools
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluate_performance.rmse import evaluate_rmse
from training.train_one_epoch import train_one_epoch
from dataloaders.load_ecotox import load_ecotox_data  
from ecotox_datasets.fingerprints_dataset import Dataset_with_fp
from models.A1_MLP_reduce_fp import FingerprintReduceMLP
from ecotox_datasets.fingerprints_dataset import Dataset_with_fp
from models.A1_MLP_reduce_fp import FingerprintReduceMLP
# -----------------------------------------------------------------------------
# 3. Main script: Data loading, grid search, training and evaluation.
# -----------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    results_file = "A1_fp_modular.csv"
    
    adore_path = '/home/tad/Desktop/Thesis files/ThesisCode/ecotox-toolkit/data_files/ecotox_mortality_processed.csv'
    chemicals_path = '/home/tad/Desktop/Thesis files/ThesisCode/ecotox-toolkit/data_files/ecotox_properties_with-oecd-function.csv'
    fingerprints_path = '/home/tad/Desktop/Thesis files/ThesisCode/ecotox-toolkit/data_files/fingerprints.csv'  # Fingerprints CSV file
    
    # Load ecotox data with fingerprint merging:
    # Note: use_fingerprint is set to True and the fingerprint column is assumed to be "morgan_fp".
    data, y = load_ecotox_data(
        adore_path=adore_path,
        chemicals_path=chemicals_path,
        use_fingerprint=True,
        fingerprint_path=fingerprints_path,
        fp_col="morgan_fp"
    )
    
    # Extract species IDs and cast them as categorical.
    species_ids = data['species'].cat.codes.values
    n_species = len(data['species'].cat.categories)
    
    # Extract the fingerprint embeddings.
    # The load_ecotox_data function has merged the fingerprint data into the DataFrame.
    # Convert the column (which stores lists as strings or already as lists) to a 2D numpy array.
    fp_embeds = np.array(list(data["morgan_fp"]))
    
    # Get the duration column.
    durations = data['duration'].values.reshape(-1, 1)
    
    # Group K-Fold split based on CAS (chemical identifier)
    groups = data['CAS'].cat.codes
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    fold_splits = list(gkf.split(fp_embeds, y, groups=groups))
    
    # -----------------------------------------------------------------------------
    # Define the hyperparameter grid for grid search.
    # Note: fp_reduce_dim is analogous to selfies_reduce_dim / mol2vec_reduce_dim.
    # -----------------------------------------------------------------------------
    param_grid = {
        'species_emb_dim': [16],
        'fp_reduce_dim': [32],  
        'hidden_sizes': [[128, 64, 32]],
        'lr': [0.001],
        'weight_decay': [1e-4],
        'n_epochs': [1], 
        'batch_size': [32]
    }
    keys = list(param_grid.keys())
    param_combos = list(itertools.product(*param_grid.values()))
    
    # -----------------------------------------------------------------------------
    # Grid Search loop over hyperparameter combinations.
    # -----------------------------------------------------------------------------
    for combo in param_combos:
        params = dict(zip(keys, combo))
        fold_rmses = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits, start=1):
            # Create training and validation datasets.
            train_dataset = Dataset_with_fp(
                species_ids[train_idx],
                durations[train_idx],
                fp_embeds[train_idx],
                y[train_idx]
            )
            val_dataset = Dataset_with_fp(
                species_ids[val_idx],
                durations[val_idx],
                fp_embeds[val_idx],
                y[val_idx]
            )
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
            
            # Initialize the fingerprint MLP model.
            # Determine fp_dim from the shape of the fingerprint embeddings.
            fp_dim = fp_embeds.shape[1]
            model = FingerprintReduceMLP(
                n_species=n_species,
                fp_dim=fp_dim,
                species_emb_dim=params['species_emb_dim'],
                fp_reduce_dim=params['fp_reduce_dim'],
                hidden_sizes=params['hidden_sizes']
            ).to(device)
            
            optimizer = optim.Adam(
                model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay']
            )
            
            # Train model for the specified number of epochs.
            for epoch in range(params['n_epochs']):
                train_one_epoch(model, train_loader, optimizer, device)
            
            # Evaluate on the validation fold.
            fold_rmse = evaluate_rmse(model, val_loader, device)
            fold_rmses.append(fold_rmse)
        
        mean_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        print(f"Params: {params} -> mean RMSE: {mean_rmse:.4f}, std RMSE: {std_rmse:.4f}")
        
        # Save the results.
        df = pd.DataFrame([{
            **params,
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse
        }])
        df.to_csv(
            results_file,
            mode='a',
            header=not os.path.exists(results_file),
            index=False
        )

if __name__ == "__main__":
    main()
