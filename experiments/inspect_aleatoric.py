import numpy as np
import pandas as pd
from pathlib import Path

ART = Path("artifacts_final")
oof_aleatoric_var = np.load(ART / "oof_aleatoric_var_64.npy")

print(f"Shape: {oof_aleatoric_var.shape}")
print(f"Unique values: {np.unique(oof_aleatoric_var)}")
print(f"Number of unique values: {len(np.unique(oof_aleatoric_var))}")

# Load splits to check if unique values correspond to folds
import joblib
splits = joblib.load(ART / "cv_splits.pkl")
print(f"Number of folds: {len(splits)}")

for i, (tr_idx, va_idx) in enumerate(splits):
    fold_vals = oof_aleatoric_var[va_idx]
    print(f"Fold {i+1} unique values: {np.unique(fold_vals)}")
