from __future__ import annotations

import argparse
import ast
import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Model (as provided by user)
# ----------------------------
class FingerprintMLP(nn.Module):
    def __init__(self, n_species, fp_embed_dim,
                 species_emb_dim=16,
                 hidden_sizes=[128, 64, 32]):
        super().__init__()
        self.species_emb = nn.Embedding(
            num_embeddings=n_species, 
            embedding_dim=species_emb_dim
        )
        # The MLP’s input dimension = species_emb_dim + fp_embed_dim + 1 (for duration)
        mlp_input_dim = species_emb_dim + fp_embed_dim + 1

        layers = []
        for hdim in hidden_sizes:
            layers.append(nn.Linear(mlp_input_dim, hdim))
            layers.append(nn.ReLU())
            mlp_input_dim = hdim
        # Final output layer with 1 unit
        layers.append(nn.Linear(mlp_input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, species_id, duration, fp_embed):
        sp_emb = self.species_emb(species_id)  # shape: (batch_size, species_emb_dim)
        if duration.dim() == 1:
            duration = duration.unsqueeze(1)
        x = torch.cat([sp_emb, fp_embed, duration], dim=1)
        out = self.mlp(x)
        return out.squeeze(-1)


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_binary_string(s: str) -> bool:
    return isinstance(s, str) and len(s) > 0 and set(s) <= {"0", "1", " ", ","}


def parse_fp(cell) -> np.ndarray:
    """Parse a fingerprint cell into a float32 numpy array.
    Handles: list/ndarray, stringified list via ast.literal_eval, or a plain 0/1 string.
    """
    if isinstance(cell, (list, np.ndarray)):
        arr = np.asarray(cell, dtype=np.float32)
        return arr
    if isinstance(cell, str):
        s = cell.strip()
        # Try literal eval first (stringified Python list/tuple)
        try:
            obj = ast.literal_eval(s)
            arr = np.asarray(obj, dtype=np.float32)
            return arr
        except Exception:
            pass
        # Fallback: comma/space separated 0/1 or plain 010101 string
        if "," in s or " " in s:
            toks = [t for t in s.replace(" ", "").split(",") if t != ""]
            return np.asarray([float(int(t)) for t in toks], dtype=np.float32)
        if set(s) <= {"0", "1"}:
            return np.asarray([float(int(ch)) for ch in s], dtype=np.float32)
    raise ValueError(f"Cannot parse fingerprint cell of type {type(cell)}: {repr(cell)[:120]}")


def parse_duration(cell) -> float:
    """Parse duration; supports numeric, '24h', '96 h', '7d', '14 d', etc. Returns float hours."""
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return np.nan
    # Numeric already
    try:
        return float(cell)
    except Exception:
        pass
    # Strings like '24h', '96 h', '7d'
    s = str(cell).strip().lower()
    # Extract number and unit
    num = ''
    unit = ''
    for ch in s:
        if ch.isdigit() or ch == '.':
            num += ch
        elif ch.isalpha():
            unit += ch
    try:
        val = float(num)
    except Exception:
        return np.nan
    if unit.startswith('h'):
        return val
    if unit.startswith('d'):
        return val * 24.0
    if unit.startswith('w'):
        return val * 24.0 * 7.0
    # Unknown unit -> assume hours
    return val


@dataclass
class Encoders:
    species2id: dict


class EcotoxFingerprintDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        species_col: str,
        duration_col: str,
        fp_col: str,
        target_col: str,
        enc: Encoders,
        duration_mean: float,
        duration_std: float,
        device: torch.device,
    ):
        self.device = device
        # Map species to ids
        self.species_ids = df[species_col].map(enc.species2id).astype(np.int64).values
        # Duration -> float hours -> z-score
        dur_raw = df[duration_col].apply(parse_duration).astype(float).values
        if not np.isfinite(dur_raw).all():
            raise ValueError("Non-finite durations after parsing; please clean duration column.")
        dur_z = (dur_raw - duration_mean) / (duration_std + 1e-8)
        self.duration = dur_z.astype(np.float32)
        # Fingerprints
        fps = [parse_fp(x) for x in df[fp_col].values]
        self.fp = np.stack(fps).astype(np.float32)
        # Target
        y = df[target_col].astype(float).values
        if not np.isfinite(y).all():
            raise ValueError("Non-finite targets; please clean target column.")
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sp = torch.tensor(self.species_ids[idx], dtype=torch.long, device=self.device)
        du = torch.tensor(self.duration[idx], dtype=torch.float32, device=self.device)
        fp = torch.tensor(self.fp[idx], dtype=torch.float32, device=self.device)
        y  = torch.tensor(self.y[idx], dtype=torch.float32, device=self.device)
        return sp, du, fp, y


# ----------------------------
# Training / Evaluation
# ----------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_one_epoch(model, loader, optimizer, criterion, max_grad_norm: Optional[float] = None):
    model.train()
    total_loss = 0.0
    n = 0
    for sp, du, fp, y in loader:
        optimizer.zero_grad(set_to_none=True)
        pred = model(sp, du, fp)
        loss = criterion(pred, y)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        bs = y.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        n += bs
    return total_loss / max(1, n)


def evaluate(model, loader, criterion) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    preds = []
    trues = []
    with torch.no_grad():
        for sp, du, fp, y in loader:
            pred = model(sp, du, fp)
            loss = criterion(pred, y)
            bs = y.shape[0]
            total_loss += float(loss.detach().cpu()) * bs
            n += bs
            preds.append(pred.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())
    preds = np.concatenate(preds) if preds else np.array([])
    trues = np.concatenate(trues) if trues else np.array([])
    val_rmse = rmse(trues, preds) if len(preds) else float('nan')
    return total_loss / max(1, n), val_rmse


# ----------------------------
# Main
# ----------------------------

def infer_fp_dim(df: pd.DataFrame, fp_col: str) -> int:
    for x in df[fp_col].values:
        if x is None:
            continue
        try:
            return parse_fp(x).shape[0]
        except Exception:
            continue
    raise RuntimeError("Could not infer fingerprint dimension from any row.")


def build_species_encoder(series: pd.Series) -> Encoders:
    cats = series.astype(str).unique().tolist()
    cats.sort()
    species2id = {s: i for i, s in enumerate(cats)}
    return Encoders(species2id=species2id)


def main():
    parser = argparse.ArgumentParser(description="Run FingerprintMLP on CSV data with K-fold CV")
    parser.add_argument('--csv', type=str, required=True, help='Path to input CSV with species, duration, fingerprint, target.')
    parser.add_argument('--target_col', type=str, default='result_conc1_mean_mol_log')
    parser.add_argument('--species_col', type=str, default='species')
    parser.add_argument('--duration_col', type=str, default='duration')
    parser.add_argument('--fp_col', type=str, default='morgan_fp')

    parser.add_argument('--species_emb_dim', type=int, default=16)
    parser.add_argument('--hidden', type=int, nargs='+', default=[128, 64, 32])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--k_folds', type=int, default=3)
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    df = pd.read_csv(args.csv)
    required_cols = [args.species_col, args.duration_col, args.fp_col, args.target_col]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in CSV. Available: {list(df.columns)[:20]} ...")

    # Infer dimensions & encoders
    fp_dim = infer_fp_dim(df, args.fp_col)
    enc = build_species_encoder(df[args.species_col])
    n_species = len(enc.species2id)
    print(f"n_species={n_species} | fp_dim={fp_dim}")

    # K-fold CV
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    fold_metrics: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df), start=1):
        print("-" * 80)
        print(f"Fold {fold}/{args.k_folds}")
        print("-" * 80)
        df_tr = df.iloc[train_idx].reset_index(drop=True)
        df_va = df.iloc[val_idx].reset_index(drop=True)

        # Duration normalization from training fold
        dur_tr = df_tr[args.duration_col].apply(parse_duration).astype(float).values
        if not np.isfinite(dur_tr).all():
            raise ValueError("Training durations contain NaNs after parsing; please clean.")
        d_mu, d_sd = float(dur_tr.mean()), float(dur_tr.std())
        if d_sd == 0.0:
            d_sd = 1.0
        
        # Datasets
        ds_tr = EcotoxFingerprintDataset(
            df_tr, args.species_col, args.duration_col, args.fp_col, args.target_col,
            enc, d_mu, d_sd, device
        )
        ds_va = EcotoxFingerprintDataset(
            df_va, args.species_col, args.duration_col, args.fp_col, args.target_col,
            enc, d_mu, d_sd, device
        )

        # Dataloaders (pin_memory not needed when tensors already on device in Dataset)
        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, drop_last=False)

        # Model / Optim
        model = FingerprintMLP(
            n_species=n_species,
            fp_embed_dim=fp_dim,
            species_emb_dim=args.species_emb_dim,
            hidden_sizes=args.hidden,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.MSELoss()

        best_rmse = float('inf')
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, dl_tr, optimizer, criterion, args.max_grad_norm)
            va_loss, va_rmse = evaluate(model, dl_va, criterion)
            print(f"Epoch {epoch:02d} | train MSE={tr_loss:.5f} | val MSE={va_loss:.5f} | val RMSE={va_rmse:.5f}")

            if va_rmse < best_rmse - 1e-6:
                best_rmse = va_rmse
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping (patience={args.patience}). Best val RMSE={best_rmse:.5f}")
                    break

        # Restore best and finalize metrics
        if best_state is not None:
            model.load_state_dict(best_state)
        _, final_rmse = evaluate(model, dl_va, criterion)
        print(f"Fold {fold} final RMSE: {final_rmse:.5f}")
        fold_metrics.append(final_rmse)

        # Optionally save fold checkpoint
        ckpt_path = f"fingerprint_mlp_fold{fold}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'n_species': n_species,
            'fp_dim': fp_dim,
            'species_emb_dim': args.species_emb_dim,
            'hidden_sizes': args.hidden,
            'species2id': enc.species2id,
            'duration_mean': d_mu,
            'duration_std': d_sd,
            'target_col': args.target_col,
            'species_col': args.species_col,
            'duration_col': args.duration_col,
            'fp_col': args.fp_col,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    # Summary
    if fold_metrics:
        mu = float(np.mean(fold_metrics))
        sd = float(np.std(fold_metrics))
        print("=" * 80)
        print(f"K={args.k_folds} | RMSE mean ± sd: {mu:.5f} ± {sd:.5f}")
        print("=" * 80)


if __name__ == '__main__':
    main()
