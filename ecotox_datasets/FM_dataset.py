import torch
from torch.utils.data import Dataset
import scipy.sparse
import numpy as np

class EcotoxFMDataset(Dataset):
    def __init__(self, X_sparse, y):
        X_dense = X_sparse.toarray()
        self.X = torch.from_numpy(X_dense).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


