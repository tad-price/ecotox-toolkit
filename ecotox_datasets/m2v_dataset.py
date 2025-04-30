import torch
from torch.utils.data import Dataset

class Mol2VecCrossDataset(Dataset):
    """
    Returns (species_id, duration, mol2vec_embed, y) for each sample.
    Exactly mirrors the SELFIES approach, except mol2vec instead of selfies.
    """
    def __init__(self, species_ids, durations, mol2vec_embeds, y):
        self.species_ids = torch.tensor(species_ids, dtype=torch.long)
        self.durations = torch.tensor(durations, dtype=torch.float)
        self.mol2vec_embeds = torch.tensor(mol2vec_embeds, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.species_ids[idx],
            self.durations[idx],
            self.mol2vec_embeds[idx],
            self.y[idx],
        )

class Dataset_with_mol2vec(torch.utils.data.Dataset):
    """
    Returns (species_ids, durations, mol2vec_embeds, y) for each sample.
    """
    def __init__(self, species_ids, durations, mol2vec_embeds, y):
        self.species_ids = torch.tensor(species_ids, dtype=torch.long)
        self.durations = torch.tensor(durations, dtype=torch.float)
        self.mol2vec_embeds = torch.tensor(mol2vec_embeds, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.species_ids[idx],
            self.durations[idx],
            self.mol2vec_embeds[idx],
            self.y[idx]
        )
