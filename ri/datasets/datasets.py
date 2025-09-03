import torch
from torch.utils.data import Dataset

class RIDataset(Dataset):
    def __init__(self, pairs):

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        x = torch.tensor(pair[0]).float()
        y = torch.tensor(pair[1]).float()

        return x, y

