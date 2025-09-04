import torch
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self, x, y, transform=None):

        assert len(x) == len(y)

        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = int(self.y[idx])

        # convert to float tensor and add channel dim
        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        if self.transform:
            img = self.transform(img)

        return img, label


class FlatMNISTDataset(Dataset):
    def __init__(self, x, y, transform=None):

        assert len(x) == len(y)

        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = int(self.y[idx])

        # convert to float tensor 
        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        if self.transform:
            img = self.transform(img)

        flat_image = img.view(-1)
        
        return flat_image, label, img
