import matplotlib.pyplot as plt
import numpy as np
import collections

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
