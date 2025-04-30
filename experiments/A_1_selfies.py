import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
from dataloaders.load_ecotox import load_ecotox_data
import numpy as np
import torch.optim as optim
import ast
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import itertools
from ecotox_datasets.selfies_dataset import SelfiesDataset
from torch.utils.data import DataLoader
from evaluate_performance.rmse import evaluate_rmse
from models.A1_MLP_reduce_selfies import SelfiesReduceMLP
from training.train_one_epoch import train_one_epoch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    results_file = "A1_selfies.csv"
    print(os.getcwd())
    data, y = load_ecotox_data(adore_path= '/home/tad/Desktop/Thesis files/ThesisCode/ecotox-toolkit/data_files/ecotox_mortality_processed.csv',
                                chemicals_path='/home/tad/Desktop/Thesis files/ThesisCode/ecotox-toolkit/data_files/ecotox_properties_with-oecd-function.csv',
                                selfies_path='/home/tad/Desktop/Thesis files/ThesisCode/ecotox-toolkit/data_files/selfies_embeddings.csv',
                                use_selfies=True)
    # Encode species as IDs
    species_ids = data['species'].cat.codes.values
    n_species = len(data['species'].cat.categories)
    
    # Convert selfies embedding columns from string to list
    data['selfies_embed'] = data['selfies_embed'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    selfies_embeds = np.array(list(data['selfies_embed']))
    selfies_embed_dim = selfies_embeds.shape[1]
    
    # Durations
    durations = data['duration'].values.reshape(-1, 1)
    
    # Group K-Fold by CAS (chemical)
    groups = data['CAS'].cat.codes
    gkf = GroupKFold(n_splits=5)
    fold_splits = list(gkf.split(selfies_embeds, y, groups=groups))

    # Define parameter grid
    param_grid = {
        'species_emb_dim': [16],
        'selfies_reduce_dim': [16],   
        'hidden_sizes': [[128, 64, 32]], 
        'lr': [0.001],
        'weight_decay': [1e-4],
        'n_epochs': [3],
        'batch_size': [32]
    }

    keys = list(param_grid.keys())
    param_combos = list(itertools.product(*param_grid.values()))
    
    # Grid Search
    for combo in param_combos:
        params = dict(zip(keys, combo))

        fold_rmses = []
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits, start=1):
            train_dataset = Dataset_with_selfies(
                species_ids[train_idx],
                durations[train_idx],
                selfies_embeds[train_idx],
                y[train_idx]
            )
            val_dataset = Dataset_with_selfies(
                species_ids[val_idx],
                durations[val_idx],
                selfies_embeds[val_idx],
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
            
            # Create model with relevant constructor params
            model = SelfiesReduceMLP(
                n_species=n_species,
                selfies_embed_dim=selfies_embed_dim,
                species_emb_dim=params['species_emb_dim'],
                selfies_reduce_dim=params['selfies_reduce_dim'],
                hidden_sizes=params['hidden_sizes'],
            ).to(device)

            # Define optimizer using the hyperparameters from params
            optimizer = optim.Adam(
                model.parameters(), 
                lr=params['lr'], 
                weight_decay=params['weight_decay']
            )
            
            # Train for n_epochs
            for epoch in range(params['n_epochs']):
                train_one_epoch(model, train_loader, optimizer, device)

            # Evaluate on validation fold
            fold_rmse = evaluate_rmse(model, val_loader, device)
            fold_rmses.append(fold_rmse)

        mean_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        print(mean_rmse, std_rmse)
        # Save results for this hyperparameter combo
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
