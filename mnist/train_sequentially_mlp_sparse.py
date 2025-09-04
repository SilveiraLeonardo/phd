from utils.utils import plot_random_digits, plot_batch_with_preds, plot_tsne, plot_batch_with_topk_probs, load_mnist_and_generate_splits, plot_accuracies, plot_weight_norm, plot_color_map
from models.models import MLPSparse 
from datasets.datasets import FlatMNISTDataset

import numpy as np
from sklearn.manifold import TSNE

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

train_ds = FlatMNISTDataset(x_train, y_train, transform)
val_ds = FlatMNISTDataset(x_val, y_val, transform)
test_ds = FlatMNISTDataset(x_test, y_test, transform)

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

model = MLPSparse()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

lambda_l1 = 1.0
print(f"lambda L1: {lambda_l1}")

# keep track of the classes seen so far
seen = []
previous = None

# start task_id from 1
for task_id, task_classes in enumerate(tasks, 1):

    print(f"task {task_id}, {task_classes}")
    
    seen += task_classes

    # train loader only uses two classes at a time
    train_loader = make_task_loader(train_ds, task_classes, batch_size, shuffle=True)
    # val and test loaders uses all classes seen so far
    val_loader = make_task_loader(val_ds, seen, batch_size, shuffle=False)
    test_loader = make_task_loader(test_ds, seen, batch_size, shuffle=False)

    if previous is not None:
        curr_loader = make_task_loader(val_ds, task_classes, batch_size, shuffle=False)
        prev_loader = make_task_loader(val_ds, previous, batch_size, shuffle=False)

    for epoch in range(epochs):
    
        model.train()
        tot_loss, tot_correct, tot_samples = 0.0, 0, 0

        curr_acc, prev_acc = [], []
        layer_stats = {}

        for xb, yb, _ in train_loader:

            optimizer.zero_grad()
            
            logits, (h1, h2) = model(xb)
            base_loss = criterion(logits, yb)

            # compute the l1 norm for the activations
            l1_norm = (h1.abs().mean() + h2.abs().mean())

            loss = base_loss + lambda_l1 * l1_norm

            loss.backward()
            optimizer.step()

            tot_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim = 1)
            tot_correct += (preds == yb).sum().item()
            tot_samples += xb.size(0)

            # in the first epoch, plot acc of current and previous task on a batch basis
            if previous is not None and epoch == 0 and len(curr_acc) <= 30:
                
                with torch.no_grad():

                    correct, n_samples = 0, 0

                    for xb, yb, _ in curr_loader:
                        logits, _ = model(xb)
                        preds = logits.argmax(dim = 1)
                        correct += (preds == yb).sum().item()
                        n_samples += xb.size(0)

                    curr_acc.append(correct / n_samples)

                    correct, n_samples = 0, 0
                    
                    for xb, yb, _ in prev_loader:
                        logits, _ = model(xb)
                        preds = logits.argmax(dim = 1)
                        correct += (preds == yb).sum().item()
                        n_samples += xb.size(0)
                        
                    prev_acc.append(correct / n_samples)
                
                for name, param in model.named_parameters():
                    if param.ndim > 1: # ignore the bias terms
                        update = -optimizer.param_groups[0]['lr'] * param.grad
                        update_norm = update.norm(2).item()
                        weight_norm = param.data.norm(2).item()

                        relative_norm = update_norm / (weight_norm + 1e+12)

                        norm_history = layer_stats.get(name, [])
                        norm_history.append(relative_norm)
                        layer_stats[name] = norm_history 

        if len(curr_acc) > 0 and len(prev_acc) > 0:
            plot_accuracies(curr_acc, prev_acc)
            plot_weight_norm(layer_stats)

        train_loss = tot_loss / tot_samples
        train_acc = tot_correct / tot_samples

        # validate on all seen classes so far
        model.eval()
        
        val_loss, val_correct, val_samples = 0.0, 0, 0
        
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                logits, _ = model(xb)
                loss = criterion(logits, yb)
                
                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim = 1)
                val_correct += (preds == yb).sum().item()
                val_samples += xb.size(0)

        val_loss /= val_samples
        val_acc = val_correct / val_samples
        
        print(f"{epoch}, train loss {train_loss:4f}, train acc {train_acc:4f}, val loss {val_loss:4f}, val acc {val_acc:4f}")

    xb, yb, xb_images = next(iter(test_loader))
    with torch.no_grad():
        logits, _ = model(xb)
        preds = logits.argmax(1)
        probs = F.softmax(logits, dim=1)
        #plot_batch_with_preds(xb, yb, preds, normalize_mean, normalize_std)
        plot_batch_with_topk_probs(xb_images, yb, preds, probs, normalize_mean, normalize_std)

    model.eval()
    all_latents = []
    all_labels  = []

    total_size = 0
    num_zeros = 0

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
    
    # the current task becomes now the previous task
    previous = task_classes




