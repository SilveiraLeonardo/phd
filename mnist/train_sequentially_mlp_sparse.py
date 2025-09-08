from utils.utils import plot_random_digits, plot_batch_with_preds, plot_tsne, plot_batch_with_topk_probs, load_mnist_and_generate_splits, plot_accuracies, plot_weight_norm, plot_color_map, plot_histogram, plot_three_color_map, plot_two_histograms, plot_class_strength, print_representation_strength_table
from models.models import MLPSparse, MLPSparseBN 
from datasets.datasets import FlatMNISTDataset

import numpy as np
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
#model = MLPSparseBN()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

lambda_l1 = 0.0
print(f"lambda L1: {lambda_l1}")

# keep track of the classes seen so far
seen = []
previous = None

representation_strength = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
}
seen_counts = []
classifier_acc = []

# start task_id from 1
for task_id, task_classes in enumerate(tasks, 1):

    print(f"task {task_id}, {task_classes}")
    
    seen += task_classes

    # train loader only uses two classes at a time
    train_loader = make_task_loader(train_ds, task_classes, batch_size, shuffle=True)
    # val and test loaders uses all classes seen so far
    val_loader = make_task_loader(val_ds, seen, batch_size, shuffle=False)
    test_loader = make_task_loader(test_ds, seen, batch_size, shuffle=False)

    one_loader = make_task_loader(val_ds, [1], batch_size, shuffle=False)
    
    if previous is not None:
        curr_loader = make_task_loader(val_ds, task_classes, batch_size, shuffle=False)
        prev_loader = make_task_loader(val_ds, previous, batch_size, shuffle=False)

    n_updates = 0
    grad_integration_list = []
    for param in model.parameters():
        if param.ndim == 2:
            grad_integration_list.append(torch.zeros_like(param.data))
    
    for epoch in range(epochs):
    
        model.train()
        tot_loss, tot_correct, tot_samples = 0.0, 0, 0

        curr_acc, prev_acc = [], []
        layer_stats = {}


        for xb, yb, _ in train_loader:

            n_updates += 1

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


            i = 0
            for param in model.parameters():
                if param.ndim == 2:
                    grad_integration_list[i] += param.grad
                    i += 1


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
        
        if train_acc > 0.98:
            print("Accuracy larger than 0.98, breaking from training...")
            break


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
    
    weights = []
    biases = []
    biases1 = []
    biases2 = []

    idx = 0
    #for param in model.parameters():
    for name, param in model.named_parameters():
        
        if (param.ndim == 2):
            weights += param.detach().cpu().view(-1).numpy().tolist()

        if (param.ndim == 1):

            if name.find("weight") != -1:
                print("bn weight, skipping")
                continue

            idx += 1

            layer_bias = param.detach().cpu().numpy().tolist()
            biases += layer_bias

            if len(biases1) == 0:
                biases1 += layer_bias
            else:
                biases2 += layer_bias

            #plot_histogram(layer_bias,
            #                    bins=40,
            #                    density=False,
            #                    y_log=False,
            #                    color='C2',
            #                    show_median=False,
            #                    title=f"Bias Distribution, layer {idx}")

    #bias_std = np.std(np.array(biases))
    #plot_histogram(biases,
    #                    bins=40,
    #                    density=False,
    #                    y_log=False,
    #                    color='C2',
    #               title=f"Bias Distribution - task {task_id} - std {bias_std:.4f}")

    bias1_std = np.std(np.array(biases1))
    bias2_std = np.std(np.array(biases2))
    #plot_two_histograms(biases1, biases2,
    #                    bins=40,
    #                    density=False,
    #                    y_log=False,
    #               titles=(f"Layer 1 - Bias Distribution - task {task_id} - std {bias1_std:.4f}",
    #                    f"Layer 2 - Bias Distribution - task {task_id} - std {bias2_std:.4f}"))
    
    weights_std = np.std(np.array(weights))
    #plot_histogram(weights,
    #                    bins=40,
    #                    density=False,
    #                    y_log=False,
    #                    color='C2',
    #               title=f"Weight Distribution - task {task_id} - std {weights_std:.4f}",
    #                    xlabel="Weight value"
    #               )

    # the current task becomes now the previous task
    previous = task_classes

    #for idx in range(len(grad_integration_list)):
    #    grad_integration_list[idx] /= n_updates

    # plot the gradients colormaps
    plot_three_color_map(grad_integration_list[0], grad_integration_list[1], grad_integration_list[2])

    #### test the strength of the representation
    # compute total number of examples in val set
    total = len(val_loader.dataset) # 2048 for first task

    # get feature size
    xb, yb, _, next(iter(val_loader))
    _, latent = model(xb)
    feat_dim = latent[1].shape[1] # 84

    # preallocate
    all_repr = np.zeros((total, feat_dim), dtype=np.float32)
    all_labels = np.zeros(total, dtype=np.int64)

    print(all_repr.shape)

    ptr = 0
    with torch.no_grad():
        for xb, yb, _ in val_loader:
            bs = xb.size(0)
            _, latent = model(xb)
            arr = latent[1].cpu().numpy()

            all_repr[ptr:ptr+bs] = arr
            all_labels[ptr:ptr+bs] = yb.cpu().numpy()
            ptr += bs

    #print(" ")
    #for c in seen:
    #    m_all = all_repr.mean(axis=0)
    #    std_all = all_repr.std(axis=0)
    #    m_class = all_repr[all_labels==c].mean(axis=0)
    #    saliency = (m_class - m_all)/(std_all + 1e-9)
    #    strength = np.linalg.norm(saliency)
    #
    #   print(f"Class {c}, strength: {strength:.4f}")

    print("Using linear probing...")
    x = all_repr
    y = all_labels
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=1000,
        tol=1e-4,
        n_jobs=-1
    )
    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)
    print(f"Linear probe overall acc: {acc:.4f}")
    classifier_acc.append(acc)

    # use the coefficients of the linear classifier to compute per-class strength
    #W = clf.coef_ # shape (n_classes, feat_dim)

    # in the binary case, the strenth of both classes is the same
    #if W.shape[0] == 1:
    #    w_c = W[0]
    #    strength = np.linalg.norm(w_c)
    #    for c in seen:
    #        print(f"Class {c}, strength, coeff weight norm: {strength:.4f}")
    #else:
    #    for i, c in enumerate(seen):
    #        w_c = W[i]
    #        strength = np.linalg.norm(w_c)
    #        print(f"Class {c}, strength, coeff weight norm: {strength:.4f}")

    
    y_pred = clf.predict(x_test)

    seen_counts.append(len(seen))
    for c in seen:
        mask = (y_test == c)
        if mask.sum() > 0:
            acc_c = (y_pred[mask] == y_test[mask]).mean()
            print(f"Class {c} accuracy on linear probing: {acc_c:.4f}")
            
            representation_strength[c].append(acc_c)

    plot_class_strength(representation_strength, seen_counts)
    print_representation_strength_table(representation_strength, classifier_acc, len(classifier_acc))

    print("")
