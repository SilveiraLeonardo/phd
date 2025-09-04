from utils.utils import plot_random_digits, plot_batch_with_preds, load_mnist_and_generate_splits, plot_tsne
from models.models import MLP 
from datasets.datasets import FlatMNISTDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
from sklearn.manifold import TSNE

seed = 42

# parameters
lr = 1e-3
epochs = 5 #10
n_val = 10000 # size of validation set
normalize_mean = 0.1307
normalize_std = 0.3081

x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_and_generate_splits(n_val=n_val, seed=seed)

plot_random_digits(x_train, y_train)

# normalize each channel of the image
transform = transforms.Compose([
    transforms.Normalize((normalize_mean,), (normalize_std,))
])

train_ds = FlatMNISTDataset(x_train, y_train, transform)
val_ds = FlatMNISTDataset(x_val, y_val, transform)
test_ds = FlatMNISTDataset(x_test, y_test, transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

model = MLP()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# training loop
for epoch in range(1, epochs):
    # train
    model.train()
    correct = 0
    for x, y, _ in train_loader:
        optimizer.zero_grad()
        logits, z = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
    
    train_acc = correct / len(train_ds)

    # validate
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y, _ in val_loader:
            logits, z = model(x)
            val_loss = criterion(logits, y)

            pred = logits.argmax(1)
            correct += (pred == y).sum().item()

    val_acc = correct / len(val_ds)

    print(f'Epoch {epoch}, train loss: {loss:.4f}, val loss: {val_loss:.4f}, train acc: {train_acc:.4f}, val acc: {val_acc:.4f}')

    xb, yb, xb_imgs = next(iter(test_loader))
    with torch.no_grad():
        logits, z = model(xb)
        preds = logits.argmax(1)
        plot_batch_with_preds(xb_imgs, yb, preds, normalize_mean, normalize_std)

model.eval()
all_latents = []
all_labels  = []

with torch.no_grad():
    for xb, yb, _ in test_loader:
        logits, z = model(xb)          # z has shape [batch,84]
        all_latents.append(z.cpu().numpy())
        all_labels .append(yb.cpu().numpy())

all_latents = np.concatenate(all_latents, axis=0)   # [N,84]
all_labels  = np.concatenate(all_labels,  axis=0)   # [N,]

tsne = TSNE(n_components=2, init='pca', random_state=42)
latents_2d = tsne.fit_transform(all_latents)  # [N,2]

plot_tsne(latents_2d, all_labels, title="MNIST latent space (2D)")
