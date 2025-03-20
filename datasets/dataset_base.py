# datasets/dataset_base.py
import torch
from torch.utils.data import Dataset

class DatasetBase(Dataset):
    def __init__(self):
        self.data = self.load_data()
        self.preprocess()

    def load_data(self):
        raise NotImplementedError

    def preprocess(self):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError
