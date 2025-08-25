import os, time, itertools, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import GroupKFold

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluate_performance.rmse import evaluate_rmse
from training.train_one_epoch import train_one_epoch
from dataloaders.load_ecotox import load_ecotox_data
from ecotox_datasets.fingerprints_dataset import Dataset_with_fp
from models.A1_MLP_reduce_fp import FingerprintReduceMLP


def parse_bracketed_float_series(fp_series: pd.Series) -> np.ndarray:
    """
    Parse strings like "[0.0, 0.0, 0.693..., ...]" into a 2D float32 array.
    """
    s = fp_series.astype("string").to_numpy()
    ex = next((x for x in s if isinstance(x, str) and len(x) > 2), None)
    if ex is None:
        raise ValueError("All fingerprints are missing/empty.")

    if not (ex[0] == '[' and ex[-1] == ']'):
        raise ValueError("Expected bracketed comma-separated lists for fingerprints.")

    arrs = [np.fromstring(x[1:-1], sep=',', dtype=np.float32) for x in s]
    width = arrs[0].size
    if any(a.size != width for a in arrs):
        raise ValueError("Inconsistent fingerprint vector lengths across rows.")
    X = np.vstack(arrs)  # float32 already
    return X


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    seed_everything(int(os.getenv("SEED", "42")))
    torch.set_num_threads(min(8, os.cpu_count() or 8))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    results_file = os.getenv("RESULTS_CSV", "A1_fp_grid.csv")

    adore_path = os.getenv("ADORE_PATH", "/home/tad/Desktop/Thesisfiles/ThesisCode/ecotox-toolkit/data_files/ecotox_mortality_processed.csv")
    chemicals_path = os.getenv("CHEM_PATH", "/home/tad/Desktop/Thesisfiles/ThesisCode/ecotox-toolkit/data_files/ecotox_properties_with-oecd-function.csv")
    fingerprints_path = os.getenv("FP_PATH", "/home/tad/Desktop/Thesisfiles/ThesisCode/ecotox-toolkit/data_files/fingerprints.csv")
    fp_col = os.getenv("FP_COL", "morgan_fp")

    n_folds = int(os.getenv("N_FOLDS", "5"))
    n_epochs_max = int(os.getenv("N_EPOCHS_MAX", "30"))
    patience = int(os.getenv("PATIENCE", "5"))
    eval_every = int(os.getenv("EVAL_EVERY", "1"))
    n_debug = int(os.getenv("N_DEBUG", "0"))       # 0 means use full data
    max_trials = int(os.getenv("MAX_TRIALS", "0")) # 0 means full grid
    num_workers = int(os.getenv("NUM_WORKERS", "4"))

    # --- load + merge ---
    t0 = time.time()
    data, y = load_ecotox_data(
        adore_path=adore_path,
        chemicals_path=chemicals_path,
        use_fingerprint=True,
        fingerprint_path=fingerprints_path,
        fp_col=fp_col
    )
    print(f"[TIMING] load_ecotox_data: {time.time()-t0:.2f}s | n={len(data)}", flush=True)

    # drop missing FPs early
    if data[fp_col].isna().any():
        n_miss = int(data[fp_col].isna().sum())
        print(f"[WARN] Dropping {n_miss} rows with missing {fp_col}", flush=True)
        keep = ~data[fp_col].isna()
        data = data.loc[keep].reset_index(drop=True)
        y = y[keep.to_numpy()]

    # --- parse morgan_fp as float vectors ---
    t1 = time.time()
    fp_embeds = parse_bracketed_float_series(data[fp_col])
    print(f"[TIMING] parse_fps: {time.time()-t1:.2f}s | shape={fp_embeds.shape} dtype={fp_embeds.dtype}", flush=True)
    assert fp_embeds.ndim == 2 and fp_embeds.dtype == np.float32

    # other inputs
    species_ids = data['species'].cat.codes.values
    n_species = len(data['species'].cat.categories)
    durations = data['duration'].values.reshape(-1, 1)

    # optional subsample for quick sanity
    if n_debug and len(data) > n_debug:
        species_ids = species_ids[:n_debug]
        durations = durations[:n_debug]
        fp_embeds = fp_embeds[:n_debug]
        y = y[:n_debug]
        data = data.iloc[:n_debug].reset_index(drop=True)
        print(f"[DEBUG] Subsampled to n={n_debug}", flush=True)

    # CV splits grouped by CAS
    groups = data['CAS'].cat.codes
    gkf = GroupKFold(n_splits=n_folds)
    fold_splits = list(gkf.split(fp_embeds, y, groups=groups))

    param_grid = {
        'species_emb_dim': [16, 32],
        'fp_reduce_dim': [128, 256, 512],          # wider since features are real-valued
        'hidden_sizes': [[512, 256, 128], [256, 128, 64]],
        'lr': [1e-3, 3e-4],
        'weight_decay': [1e-4, 3e-5],
        'batch_size': [int(os.getenv("BATCH_SIZE", "512"))],
    }
    keys = list(param_grid.keys())
    all_combos = [dict(zip(keys, vals)) for vals in itertools.product(*param_grid.values())]

    rng = random.Random(int(os.getenv("SAMPLE_SEED", "123")))
    if max_trials and max_trials < len(all_combos):
        rng.shuffle(all_combos)
        combos = all_combos[:max_trials]
        print(f"[GRID] Random subset {max_trials}/{len(all_combos)}", flush=True)
    else:
        combos = all_combos
        print(f"[GRID] Full grid size = {len(combos)}", flush=True)

    rows_out = []
    best_overall = (np.inf, None)

    for i, params in enumerate(combos, start=1):
        print("\n" + "-"*80, flush=True)
        print(f"[{i}/{len(combos)}] params={params}", flush=True)
        fold_rmses = []

        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits, start=1):
            print(f"\n[Fold {fold_idx}/{n_folds}]", flush=True)

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

            pin = torch.cuda.is_available()
            train_loader = DataLoader(
                train_dataset,
                batch_size=params['batch_size'],
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=max(1024, params['batch_size']),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin,
            )

            fp_dim = fp_embeds.shape[1]
            model = FingerprintReduceMLP(
                n_species=n_species,
                fp_dim=fp_dim,
                species_emb_dim=params['species_emb_dim'],
                fp_reduce_dim=params['fp_reduce_dim'],
                hidden_sizes=params['hidden_sizes'],
            ).to(device)

            optimizer = optim.Adam(
                model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay']
            )

            # early stopping on val RMSE
            best_rmse = np.inf
            no_improve = 0
            t_fold = time.time()

            for epoch in range(1, n_epochs_max + 1):
                train_one_epoch(model, train_loader, optimizer, device)
                if epoch % eval_every == 0 or epoch == n_epochs_max:
                    rmse = evaluate_rmse(model, val_loader, device)
                    print(f"  epoch {epoch:3d} | val RMSE={rmse:.4f}", flush=True)
                    if rmse + 1e-6 < best_rmse:
                        best_rmse = rmse
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            print(f"  early stop (patience {patience}) at epoch {epoch}", flush=True)
                            break

            print(f"[Fold {fold_idx}] best RMSE={best_rmse:.4f} | time={time.time()-t_fold:.2f}s", flush=True)
            fold_rmses.append(best_rmse)

        mean_rmse = float(np.mean(fold_rmses))
        std_rmse = float(np.std(fold_rmses))
        print(f"\n[RESULT] params={params} -> mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}", flush=True)

        if mean_rmse < best_overall[0]:
            best_overall = (mean_rmse, params)

        row = {
            **params,
            'n_folds': n_folds,
            'n_epochs_max': n_epochs_max,
            'patience': patience,
            'eval_every': eval_every,
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
        }
        for j, rm in enumerate(fold_rmses, 1):
            row[f'fold{j}_rmse'] = float(rm)
        rows_out.append(row)

        pd.DataFrame([row]).to_csv(
            results_file,
            mode='a',
            header=not os.path.exists(results_file),
            index=False
        )

    print("\n" + "="*80, flush=True)
    print(f"[BEST] mean RMSE={best_overall[0]:.4f} with params={best_overall[1]}", flush=True)
    print(f"[SAVED] {results_file}", flush=True)


if __name__ == "__main__":
    main()
