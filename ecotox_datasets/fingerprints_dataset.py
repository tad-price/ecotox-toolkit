import torch
from torch.utils.data import Dataset

class Dataset_with_fp(torch.utils.data.Dataset):
    """
    Returns (species_ids, durations, fp_embeds, y) for each sample.
    """
    def __init__(self, species_ids, durations, fp_embeds, y):
        self.species_ids = torch.tensor(species_ids, dtype=torch.long)
        self.durations = torch.tensor(durations, dtype=torch.float)
        self.fp_embeds = torch.tensor(fp_embeds, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.species_ids[idx],
            self.durations[idx],
            self.fp_embeds[idx],
            self.y[idx]
        )
