from utils.utils import plot_random_digits, plot_batch_with_preds, load_mnist_and_generate_splits
from models.models import LeNet5
from datasets.datasets import MNISTDataset

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

seed = 42

# parameters
lr = 1e-3
epochs = 5
batch_size = 64
n_val = 10000 # size of validation set
normalize_mean = 0.1307
normalize_std = 0.3081

x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_and_generate_splits(n_val=n_val, seed=seed)

#plot_random_digits(x_train, y_train)

# normalize each channel of the image
transform = transforms.Compose([
    transforms.Normalize((normalize_mean,), (normalize_std,))
])

train_ds = MNISTDataset(x_train, y_train, transform)
val_ds = MNISTDataset(x_val, y_val, transform)
test_ds = MNISTDataset(x_test, y_test, transform)

#train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# define the 5 class-incremental tasks
tasks = [
    [1, 2],
    [3, 4], 
    [5, 6],
    [7, 8],
    [9, 0],
]

def make_task_loader(full_ds, target_classes, batch_size=64, shuffle=True):
    # dataloader that only return samples whose labels are in target_classes
    labels = np.array(full_ds.y)

    mask = np.isin(labels, target_classes)

    # indices of examples belonging to the target classes
    idxs = np.nonzero(mask)[0].tolist()

    # take a subset of a dataset at given indices
    sub_ds = Subset(full_ds, idxs)

    return DataLoader(sub_ds, batch_size=batch_size, shuffle=shuffle)

model = LeNet5()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# keep track


