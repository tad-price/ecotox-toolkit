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
from models.A1_MLP_reduce_m2v import Mol2vecReduceMLP  
from training.train_one_epoch import train_one_epoch
from dataloaders.load_ecotox import load_ecotox_data 
from ecotox_datasets.m2v_dataset import Dataset_with_mol2vec

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    results_file = "A1_m2v_modular.csv"
    
    # File paths (adjust these paths to point to your data files)
    adore_path = '/home/tad/Desktop/Thesis files/ThesisCode/ecotox-toolkit/data_files/ecotox_mortality_processed.csv'
    chemicals_path = '/home/tad/Desktop/Thesis files/ThesisCode/ecotox-toolkit/data_files/ecotox_properties_with-oecd-function.csv'

    # For mol2vec, we assume the required columns are already in the chemicals file.
    mol2vec_cols = [f'chem_mol2vec{str(i).zfill(3)}' for i in range(300)]
    
    # Load data with use_mol2vec=True (note: mol2vec_path is None)
    data, y = load_ecotox_data(
        adore_path=adore_path,
        chemicals_path=chemicals_path,
        use_mol2vec=True,
        mol2vec_path=None,      # using columns from the chemicals file
        mol2vec_cols=mol2vec_cols
    )
    
    # Extract fields
    species_ids = data['species'].cat.codes.values
    n_species = len(data['species'].cat.categories)
    mol2vec_embeds = data[mol2vec_cols].values
    durations = data['duration'].values.reshape(-1, 1)
    
    # Set up Group K-Fold based on CAS (chemical identifier)
    groups = data['CAS'].cat.codes
    gkf = GroupKFold(n_splits=5)
    fold_splits = list(gkf.split(mol2vec_embeds, y, groups=groups))
    
    # Define parameter grid
    param_grid = {
        'species_emb_dim': [16],
        'mol2vec_reduce_dim': [16],
        'hidden_sizes': [[128, 64, 32]],
        'lr': [0.001],
        'weight_decay': [1e-4],
        'n_epochs': [1],
        'batch_size': [32]
    }
    keys = list(param_grid.keys())
    param_combos = list(itertools.product(*param_grid.values()))
    
    # Grid search over hyperparameters
    for combo in param_combos:
        params = dict(zip(keys, combo))
        fold_rmses = []
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits, start=1):
            train_dataset = Dataset_with_mol2vec(
                species_ids[train_idx],
                durations[train_idx],
                mol2vec_embeds[train_idx],
                y[train_idx]
            )
            val_dataset = Dataset_with_mol2vec(
                species_ids[val_idx],
                durations[val_idx],
                mol2vec_embeds[val_idx],
                y[val_idx]
            )
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=params['batch_size'], 
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=params['batch_size'], 
                shuffle=False
            )
            
            # Initialize model using the Mol2vecReduceMLP architecture
            model = Mol2vecReduceMLP(
                n_species=n_species,
                mol2vec_dim=mol2vec_embeds.shape[1],
                species_emb_dim=params['species_emb_dim'],
                mol2vec_reduce_dim=params['mol2vec_reduce_dim'],
                hidden_sizes=params['hidden_sizes'],
            ).to(device)
            
            optimizer = optim.Adam(
                model.parameters(), 
                lr=params['lr'], 
                weight_decay=params['weight_decay']
            )
            
            # Train for specified epochs
            for epoch in range(params['n_epochs']):
                train_one_epoch(model, train_loader, optimizer, device)
                
            fold_rmse = evaluate_rmse(model, val_loader, device)
            fold_rmses.append(fold_rmse)
        
        mean_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        print(f"Params: {params} -> mean RMSE: {mean_rmse:.4f}, std RMSE: {std_rmse:.4f}")
        
        # Append results to CSV
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
