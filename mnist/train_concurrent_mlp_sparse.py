from utils.utils import plot_random_digits, plot_batch_with_preds, load_mnist_and_generate_splits, plot_tsne, plot_color_map, plot_histogram, plot_weight_and_biases
from models.models import MLPSparse 
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
epochs = 5
n_val = 10000 # size of validation set
normalize_mean = 0.1307
normalize_std = 0.3081

x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_and_generate_splits(n_val=n_val, seed=seed)

#plot_random_digits(x_train, y_train)

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

model = MLPSparse()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
criterion = nn.CrossEntropyLoss()

lambda_l1 = 0.0
print(f"lambda L1: {lambda_l1}")

# training loop
for epoch in range(1, epochs):
    # train
    model.train()
    correct = 0
    for x, y, _ in train_loader:
        optimizer.zero_grad()
        logits, (h1, h2) = model(x)
        base_loss = criterion(logits, y)

        # compute l1 norm for all the weights
        #l1_norm = 0.0
        #for param in model.parameters():
        #    l1_norm += torch.sum(torch.abs(param))

        # compute the l1 norm for the activations
        l1_norm = (h1.abs().mean() + h2.abs().mean())

        loss = base_loss + lambda_l1 * l1_norm

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
            logits, _ = model(x)
            val_loss = criterion(logits, y)

            pred = logits.argmax(1)
            correct += (pred == y).sum().item()

    val_acc = correct / len(val_ds)

    print(f'Epoch {epoch}, train loss: {loss:.4f}, val loss: {val_loss:.4f}, train acc: {train_acc:.4f}, val acc: {val_acc:.4f}')

    xb, yb, xb_imgs = next(iter(test_loader))
    with torch.no_grad():
        logits, _ = model(xb)
        preds = logits.argmax(1)
        plot_batch_with_preds(xb_imgs, yb, preds, normalize_mean, normalize_std)

model.eval()
all_latents = []
all_labels  = []

total_size = 0
num_zeros = 0

weights = []
biases = []
idx = 0
for param in model.parameters():
    if (param.ndim == 2):
        weights += param.detach().cpu().view(-1).numpy().tolist()

    if (param.ndim == 1):

        idx += 1

        layer_bias = param.detach().cpu().numpy().tolist()
        biases += layer_bias

        #plot_histogram(layer_bias,
        #                    bins=40,
        #                    density=False,
        #                    y_log=False,
        #                    color='C2',
        #                    show_median=False,
        #                    title=f"Bias Distribution, layer {idx}")

bias_std = np.std(np.array(biases))
plot_histogram(biases,
                    bins=40,
                    density=False,
                    y_log=False,
                    color='C2',
               title=f"Bias Distribution - std {bias_std:.4f}")

weights_std = np.std(np.array(weights))
plot_histogram(weights,
                    bins=40,
                    density=False,
                    y_log=False,
                    color='C2',
               title=f"Weight Distribution - std {weights_std:.4f}",
                    xlabel="Weight value"
               )


with torch.no_grad():
    for idx, (xb, yb, _) in enumerate(test_loader):
        logits, latent = model(xb)          # z has shape [batch,84]
        all_latents.append(latent[1].cpu().numpy())
        all_labels .append(yb.cpu().numpy())

        for hidden_pt in latent:
            hidden = hidden_pt.cpu().numpy()
            total_size += np.prod(hidden.shape)
            num_zeros += (hidden == 0).sum()

        # plot colormaps only for the first batch
        if idx == 0:
            plot_color_map(latent[0], latent[1])

all_latents = np.concatenate(all_latents, axis=0)   # [N,84]
all_labels  = np.concatenate(all_labels,  axis=0)   # [N,]

tsne = TSNE(n_components=2, init='pca', random_state=42)
latents_2d = tsne.fit_transform(all_latents)  # [N,2]

plot_tsne(latents_2d, all_labels, title="MNIST latent space (2D)")

population_sparcity = (num_zeros / total_size) # total_size = n_examples * n_neurons
print(f"Sparcity analysis - population sparcity: {population_sparcity:.4f}")


# plot parameters of the neural network
model_weigths = []
model_biases = []
for name, param in model.named_parameters():
    if param.ndim == 2:
        model_weigths.append(param.data.detach().cpu().numpy())
    elif param.ndim == 1:
        if name.find("weight") != -1:
            print("bn weight, skipping... we will keep only the bn bias")
            continue
        model_biases.append(param.data.detach().cpu().numpy())
        if param.numel() == 10: # number of out classes
            print("Classification bias vector:")
            print(param)
plot_weight_and_biases(model_weigths, model_biases)
