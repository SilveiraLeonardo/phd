import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import collections
import torch

import idx2numpy

def open_dataset(partition="train"):

    if partition=="train":
        images = idx2numpy.convert_from_file('../dataset/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte')
        labels = idx2numpy.convert_from_file('../dataset/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    else:
        images = idx2numpy.convert_from_file('../dataset/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
        labels = idx2numpy.convert_from_file('../dataset/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    print(f'{partition} dataset opened correctly')
    print(f'data shape: {images.shape}, labels shape: {labels.shape}')

    return images, labels

def load_mnist_and_generate_splits(n_val=1000, seed=42):

    np.random.seed(seed)

    # parameters
    n_val = 10000 # size of validation set

    x_train_full, y_train_full = open_dataset()
    x_test, y_test = open_dataset("test")

    freq_train_labels = collections.Counter(y_train_full)
    freq_test_labels = collections.Counter(y_test)

    print(f'frequency of train labels: {freq_train_labels}')
    print(f'frequency of test labels: {freq_test_labels}')

    indices = np.random.permutation(len(x_train_full))

    train_idx = indices[n_val:] # last 50k
    val_idx = indices[:n_val] # first 10k

    x_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]

    x_train = x_train_full[train_idx]
    y_train = y_train_full[train_idx]

    print(f'splitting into train: x: {x_train.shape}, y: {y_train.shape}, and val: x: {x_val.shape}, y: {y_val.shape} datasets...')

    sample = x_train[100]
    print(f'shape of the inputs: {sample.shape}')

    return x_train, y_train, x_val, y_val, x_test, y_test

def unnormalize_batch(x, mean, std):
    return x * std + mean

def unnormalize_batch_three_channels(x, mean, std):
    # move mean/std to the same device & give them shape (1,C,1,1)
    mean = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    std  = torch.tensor(std,  device=x.device).view(1, -1, 1, 1)
    return x * std + mean

def plot_batch_with_preds(x, y_true, y_pred, mean, std, n_rows=2, n_cols=5,
                       figsize=(10, 5), cmap='gray'):

    x = unnormalize_batch(x, mean, std).cpu().numpy()

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    B = x.shape[0]
    n_images = n_rows * n_cols

    idx = np.random.choice(B, n_images, replace=False)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    for ax, i in zip(axes, idx):

        img = x[i].squeeze() # shape: 28x28
        true = y_true[i]
        pred = y_pred[i]
        color = 'green' if true == pred else 'red'

        ax.imshow(img, cmap=cmap)
        ax.set_title(f"T: {true} P: {pred}", color=color)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_batch_with_topk_probs(x, y_true, y_pred, probs, mean, std, n_cols=5, topk=5,
                       figsize=(12, 6), cmap='gray'):

    x = unnormalize_batch(x, mean, std).cpu().numpy()

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    B = x.shape[0]

    idx = np.random.choice(B, n_cols, replace=False)

    fig, axes = plt.subplots(2, n_cols, figsize=figsize,
                             gridspec_kw={'height_ratios':[1,1]})
    for col, i in enumerate(idx):
        img = x[i].squeeze()   # shape (H,W), because MNIST is single‐channel

        true = y_true[i]
        pred = y_pred[i]
        color = 'green' if true == pred else 'red'

        # ---- TOP ROW: the image ----
        ax_img = axes[0, col]
        ax_img.imshow(img, cmap=cmap)
        ax_img.set_title(f"T:{true}  P:{pred}", color=color)
        ax_img.axis('off')

        # ---- BOTTOM ROW: barplot of top‐k probs ----
        ax_bar = axes[1, col]
        # get topk indices & probs
        pi = probs[i]  # length‐10

        #print("-----")
        #print(true)
        #print(pi[true])
        #print(pi)

        topk_idx = np.argsort(pi)[-topk:].tolist()[::-1]  # descending
        topk_vals = pi[topk_idx]

        bars = ax_bar.bar(range(topk), topk_vals, color='lightblue')
        # highlight the predicted label bar in a different color
        for b, lbl in zip(bars, topk_idx):
            if lbl == pred:
                b.set_color('orange')

        ax_bar.set_xticks(range(topk))
        ax_bar.set_xticklabels(topk_idx, fontsize=10)
        ax_bar.set_ylim(0, 1.0)
        ax_bar.set_ylabel("P", fontsize=10)
        ax_bar.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_batch_with_topk_probs_three_channels(x, y_true, y_pred, probs, mean, std, n_cols=5, topk=5, figsize=(12, 6), cmap='gray'):

    #x = unnormalize_batch(x, mean, std).cpu().numpy()
    x = unnormalize_batch_three_channels(x, mean, std).cpu().numpy() # (B, 3, H, W)
    x = np.clip(x, 0.0, 1.0)

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    if torch.is_tensor(probs):
        probs = probs.cpu().numpy()

    B = x.shape[0]

    idx = np.random.choice(B, n_cols, replace=False)

    fig, axes = plt.subplots(2, n_cols, figsize=figsize,
                             gridspec_kw={'height_ratios':[1,1]})
    for col, i in enumerate(idx):
        #img = x[i].squeeze()   # shape (H,W), because MNIST is single‐channel
        img = x[i].transpose(1,2,0)   # now (H,W,3)

        true = y_true[i]
        pred = y_pred[i]
        color = 'green' if true == pred else 'red'

        # ---- TOP ROW: the image ----
        ax_img = axes[0, col]
        ax_img.imshow(img, cmap=cmap)
        ax_img.set_title(f"T:{true}  P:{pred}", color=color)
        ax_img.axis('off')

        # ---- BOTTOM ROW: barplot of top‐k probs ----
        ax_bar = axes[1, col]
        # get topk indices & probs
        pi = probs[i]  # length‐10

        topk_idx = np.argsort(pi)[-topk:].tolist()[::-1]  # descending
        topk_vals = pi[topk_idx]

        bars = ax_bar.bar(range(topk), topk_vals, color='lightblue')
        # highlight the predicted label bar in a different color
        for b, lbl in zip(bars, topk_idx):
            if lbl == pred:
                b.set_color('orange')

        ax_bar.set_xticks(range(topk))
        ax_bar.set_xticklabels(topk_idx, fontsize=10)
        ax_bar.set_ylim(0, 1.0)
        ax_bar.set_ylabel("P", fontsize=10)
        ax_bar.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_random_digits(images, labels, n_rows=2, n_cols=5,
                       figsize=(10, 5), cmap='gray', seed=None):

    rng = np.random.RandomState(seed)
    n_images = n_rows * n_cols
    idx = rng.choice(len(images), n_images, replace=False)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    for ax, i in zip(axes, idx):
        img = images[i]
        # if flat, reshape to 28×28
        if img.ndim == 1:
            img = img.reshape(28, 28)
        ax.imshow(img, cmap=cmap)
        ax.set_title(f"Label: {labels[i]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_tsne(latents_2d, all_labels, title="MNIST latent space (2D)"):

    plt.figure(figsize=(8,8))
    sc = plt.scatter(latents_2d[:,0],
                     latents_2d[:,1],
                     c=all_labels,
                     cmap='tab10',
                     s=5,
                     alpha=0.7)
    plt.legend(*sc.legend_elements(), title="classes")
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.grid(False)
    plt.show()

def plot_accuracies(line1, line2, line1_title='Current Task', line2_title='Previous Task',
                    xlabel='Batch # in Epoch 0', ylabel='Accuracy', 
                    title='Batch-wise Accuracy: Current vs. Previous Task'):

    batches = list(range(1, len(line1) + 1))

    plt.figure(figsize=(6,4))
    plt.plot(batches, line1,  marker='o', label=line1_title)
    plt.plot(batches, line2,  marker='s', label=line2_title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(batches)           # show each batch on the x-axis
    plt.ylim(0,1)                 # accuracy between 0 and 1
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_weight_norm(layer_stats):

    plt.figure(figsize=(10,6))

    for name, history in layer_stats.items():
        plt.plot(history, label=name)

    plt.yscale('log')
    plt.xlabel("Batch index")
    plt.ylabel("Relative ‖Δw‖/‖w‖ (log scale)")
    plt.title("Per-layer relative update norms over batches")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.show()

def plot_color_map(data1, data2, title1="Hidden 1", title2="Hidden 2"):

    # Heat map
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, 
        ncols=1, 
        figsize=(6,8), 
        constrained_layout=True)

    # use the same vmin and vmax colors to be comparable
    vmin = min(data1.min(), data2.min())
    vmax = max(data1.max(), data2.max())

    im1 = ax1.imshow(data1, vmin=vmin, vmax=vmax, aspect="auto")
    ax1.set_title(title1)

    im2 = ax2.imshow(data2, vmin=vmin, vmax=vmax, aspect="auto")
    ax2.set_title(title2)
    
    # Add the color bar
    cbar = fig.colorbar(im2, ax = [ax1, ax2], orientation="vertical", fraction=0.02, pad=0.04)
    cbar.ax.set_ylabel("Shared color bar", rotation = -90, va = "bottom")

    plt.show()





