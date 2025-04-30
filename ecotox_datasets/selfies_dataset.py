import torch
from torch.utils.data import Dataset, DataLoader

class SelfiesDataset(Dataset):
    def __init__(self, species_ids, durations, selfies_embeds, y):
        self.species_ids = torch.tensor(species_ids, dtype=torch.long)
        self.durations = torch.tensor(durations, dtype=torch.float)
        self.selfies_embeds = torch.tensor(selfies_embeds, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.species_ids[idx],
            self.durations[idx],
            self.selfies_embeds[idx],
            self.y[idx],
        )