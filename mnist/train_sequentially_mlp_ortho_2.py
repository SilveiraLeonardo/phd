from utils.utils import plot_random_digits, plot_batch_with_preds, plot_tsne, plot_batch_with_topk_probs, load_mnist_and_generate_splits, plot_accuracies, plot_weight_norm, plot_color_map, plot_histogram, plot_three_color_map, plot_two_histograms, plot_class_strength, print_representation_strength_table, plot_weight_and_biases
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

# orthogonal gradient descen
class OGDProjector:
    def __init__(self, model, eps=1e-8):
        self.model = model
        # list of torch tensors, each is a unit-norm flattened gradient
        self.basis = []
        self.eps = eps

    def __len__(self):
        return len(self.basis)
    
    def _flatten_grads(self):
        flats = []
        for p in self.model.parameters():
            if p.grad is not None:
                flats.append(p.grad.detach().view(-1))
        if len(flats) == 0:
            return None
        return torch.cat(flats)

    def register_task_gradients(self):
        # call after finishing a task
        g = self._flatten_grads()
        if g is None:
            return
        # Gram-Schmidt against existing basis
        for b in self.basis:
            # for each basis vector
            # subtract from g the components in the direction of b, for each b
            # so that the vectors in g only have the orthogonal components now
            g -= b * (b @ g)
        ng = g.norm()
        if ng.item() > self.eps:
            self.basis.append(g / ng)

    def project_current_gradients(self):
        # call after loss.backward() and before optimizer.step()
        grads = self._flatten_grads()
        if grads is None or len(self.basis) == 0:
            return
        # project grads onto orthogonal complement of the basis
        g = grads.clone()
        for b in self.basis:
            g -= b * (b @ g)

        # write g back into parameter gradients
        idx = 0
        for p in self.model.parameters():
            if p.grad is not None:
                numel = p.grad.numel() # total number of elements in the tensor
                new_grad = g[idx:idx+numel].view_as(p.grad)
                p.grad.data.copy_(new_grad)
                idx += numel

seed = 42

# parameters
lr = 1e-3
epochs = 15
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

#model = MLPSparse()
model = MLPSparseBN()

ogd = OGDProjector(model)

weight_decay = 1e-5 #1e-2
print(f"Weight decay: {weight_decay}")
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

#lr1 = 1e-3
#lr2 = 1e-4

#optimizer = torch.optim.AdamW([
#    { 'params': list(model.fc1.parameters()) + list(model.fc2.parameters()), 'lr': lr1, 'weight_decay': weight_decay },
#    { 'params': list(model.bn1.parameters()) + list(model.bn2.parameters()), 'lr': lr1, 'weight_decay': 0.0 },
#    { 'params': model.fc3.parameters(), 'lr': lr2, 'weight_decay': weight_decay },
#])

criterion = nn.CrossEntropyLoss()

lambda_l1 = 0.001
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

    # loader to evaluate the forward transfer ability of the network
    forward_transfer_loader = make_task_loader(val_ds, [9, 0], batch_size, shuffle=False)
    # loader to evaluate the ability of the network to evaluate all classes
    all_classes_loader = make_task_loader(val_ds, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0], batch_size, shuffle=False)
    
    if previous is not None:
        curr_loader = make_task_loader(val_ds, task_classes, batch_size, shuffle=False)
        prev_loader = make_task_loader(val_ds, previous, batch_size, shuffle=False)

    n_updates = 0
    grad_integration_list = []
    grad_biases_integration_list = []
    for name, param in model.named_parameters():
        if param.ndim == 2:
            grad_integration_list.append(torch.zeros_like(param.data))
        elif param.ndim == 1:
            if name.find("weight") != -1:
                continue
            grad_biases_integration_list.append(torch.zeros_like(param.data))

    print(f"lambda l1: {lambda_l1}")

    curr_acc, prev_acc = [], []
    show_forgetting_curve = True 
    
    for epoch in range(epochs):
    
        model.train()
        tot_loss, tot_correct, tot_samples = 0.0, 0, 0

        layer_stats = {}

        for counter, (xb, yb, _) in enumerate(train_loader):

            n = xb.size(0) # batch size

            n_updates += 1

            optimizer.zero_grad()
            
            logits, (h1, h2) = model(xb)

            #base_loss = criterion(logits, yb)
            #base_loss = -torch.log(F.sigmoid(logits[range(n), yb])).sum()/n
            
            # loss for the correct class: attracts it to 0.9
            #base_loss = (F.sigmoid(logits[range(n), yb]) - 0.9).pow(2).sum() / n
            
            # loss for the incorrect classes: attracts it to 0.1
            # do not backprop through the classification layer, only latent layers
            # 1) create fake final layer, whose weights and bias are detached
            #fake_logits = F.linear(
            #        h2, 
            #        model.fc3.weight.detach(),
            #        model.fc3.bias.detach())
            #fake_probs = F.sigmoid(fake_logits)
            # 2) mask to select only the incorrect classes
            #mask = torch.ones_like(fake_probs, dtype=torch.bool)
            #mask[range(n), yb] = False
            # 3) calculate the loss for the incorrect classes
            #repel_loss = (fake_probs[mask] - 0.1).pow(2).sum() / n

            # apply cross-entropy loss
            task_logits = logits[:, task_classes]
            tc = torch.tensor(task_classes) # tensor([1, 2])
            yb_expand = yb.unsqueeze(1) # (batch, 1)
            targets = (yb_expand == tc).float() # (batch, 2)
            base_loss = F.binary_cross_entropy_with_logits(task_logits, targets)

            if epoch == 2 and counter == 0:
                print("checking...")
                print(base_loss)
                print("example")
                print(yb[0])
                print(F.sigmoid(logits[0]))
                print(F.sigmoid(logits[0, yb[0]]))
                print((F.sigmoid(logits[0, yb[0]])-0.9).pow(2))

            # compute the l1 norm for the activations
            #l1_norm = (h1.abs().mean() + h2.abs().mean())
            l1_norm = 0.0
            for name, param in model.named_parameters():
                if name == "fc3.weight":
                    l1_norm += param[task_classes, :].pow(2).sum()
                elif name == "fc3.bias":
                    l1_norm += param[task_classes].pow(2).sum()

            #loss = base_loss + 0.1 * repel_loss + lambda_l1 * l1_norm
            loss = base_loss + lambda_l1 * l1_norm

            loss.backward()

            # OGD project the main gradient
            ogd.project_current_gradients()

            optimizer.step()

            # reduce over confidence in wrong logits
            neg_threshold = 0.5
            neg_lr = 1e-3
            
            probs = torch.sigmoid(logits)

            for c in range(10):
                if c in task_classes:
                    continue

                # find which samples in the batch are over-confident on class c
                mask = probs[:, c] > neg_threshold
                if not mask.any():
                    continue

                # zero out any existing gradient
                optimizer.zero_grad()

                # loss: sum over the selected examples
                loss_c = logits[mask, c].sum()
                loss_c.backward()

                # pull out the flattened gradients
                flat_g = ogd._flatten_grads()
                if flat_g is None:
                    continue

                # orthogonally project into your basis
                g_proj = flat_g.clone()
                for b in ogd.basis:
                    g_proj -= b * (b @ g_proj)

                # take a small negative step along g_proj
                # to decrease over-confident logits
                idx = 0
                with torch.no_grad:
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        n = p.numel()
                        g_piece = g_proj[idx:idx+n].view_as(p)
                        p.data -= neg_lr * g_piece
                        idx += n
                        
                optimizer.zero_grad()

            tot_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim = 1)
            tot_correct += (preds == yb).sum().item()
            tot_samples += xb.size(0)

            i = 0
            j = 0
            for name, param in model.named_parameters():
                
                if param.ndim == 2:
                    grad_integration_list[i] += param.grad
                    i += 1
                elif param.ndim == 1:
                    if name.find("weight") != -1:
                        continue
                    grad_biases_integration_list[j] += param.grad
                    j += 1

            # in the first epoch, plot acc of current and previous task on a batch basis
            #if previous is not None and epoch == 0 and len(curr_acc) <= 30:
            if previous is not None and counter % 10 == 0 and len(curr_acc) <= 70:
                
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
                
            if previous is not None and epoch == 0 and counter <= 30:
                for name, param in model.named_parameters():
                    if param.ndim > 1: # ignore the bias terms
                        update = -optimizer.param_groups[0]['lr'] * param.grad
                        update_norm = update.norm(2).item()
                        weight_norm = param.data.norm(2).item()

                        relative_norm = update_norm / (weight_norm + 1e+12)

                        norm_history = layer_stats.get(name, [])
                        norm_history.append(relative_norm)
                        layer_stats[name] = norm_history 

        if len(curr_acc) > 0 and len(prev_acc) > 0 and show_forgetting_curve:
            plot_accuracies(curr_acc, prev_acc)
            plot_weight_norm(layer_stats)
            if len(curr_acc) > 70:
                show_forgetting_curve = False
        #if len(curr_acc) > 0 and len(prev_acc) > 0:
        #    plot_accuracies(curr_acc, prev_acc)
        #    plot_weight_norm(layer_stats)

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

    # update basis of gradient vectors
    model.train()
    
    print("registering gradients for the task")
    running_g = {c: None for c in task_classes}
    counts = {c: 0 for c in task_classes}

    for idx, (xb, yb, _) in enumerate(train_loader):

        for c in task_classes:
            mask = (yb == c)
            if not mask.any():
                continue

            optimizer.zero_grad()
            logits, _ = model(xb)
       
            # sum of the correct class logits over the minibatch
            loss_c = logits[mask, c].sum()
            # backward
            loss_c.backward()

            flat_g = ogd._flatten_grads()
            if flat_g is not None:
                if running_g[c] is None:
                    running_g[c] = flat_g.clone()
                else:
                    running_g[c] += flat_g
                count += 1

    for c in task_classes:
        if counts[c] == 0:
            continue

        g = running_g[c] / float(counts[c])

        # Gram-Schmidt
        for b in ogd.basis:
            g -= b * (b @ g)
        ng = g.norm()
        if ng > ogd.eps:
           ogd.basis.append(g / ng)

    print(f"{len(ogd)} gradients stored")

    model.eval()

    xb, yb, xb_images = next(iter(test_loader))
    with torch.no_grad():
        logits, _ = model(xb)
        preds = logits.argmax(1)
        probs = F.softmax(logits, dim=1)
        #plot_batch_with_preds(xb, yb, preds, normalize_mean, normalize_std)
        plot_batch_with_topk_probs(xb_images, yb, preds, probs, normalize_mean, normalize_std)

    model.eval()

    print("Checking norm of the class. layer weights")
    for name, param in model.named_parameters():
        
        if name == "fc3.weight":
            print(param.data.norm(dim=1))
        elif name == "fc3.bias":
            print(param.data.norm())


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
   
    # analysis of the distribution of biases and weights
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

    #bias1_std = np.std(np.array(biases1))
    #bias2_std = np.std(np.array(biases2))
    #plot_two_histograms(biases1, biases2,
    #                    bins=40,
    #                    density=False,
    #                    y_log=False,
    #               titles=(f"Layer 1 - Bias Distribution - task {task_id} - std {bias1_std:.4f}",
    #                    f"Layer 2 - Bias Distribution - task {task_id} - std {bias2_std:.4f}"))
    
    #weights_std = np.std(np.array(weights))
    #plot_histogram(weights,
    #                    bins=40,
    #                    density=False,
    #                    y_log=False,
    #                    color='C2',
    #               title=f"Weight Distribution - task {task_id} - std {weights_std:.4f}",
    #                    xlabel="Weight value"
    #               )

    # analysis of the accumulation of gradients

    # if dont want the gradient accumulated, but the mean value
    #for idx in range(len(grad_integration_list)):
    #    grad_integration_list[idx] /= n_updates

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
    
    # plot the gradients colormaps
    plot_weight_and_biases(grad_integration_list, grad_biases_integration_list)
    #plot_three_color_map(grad_integration_list[0], grad_integration_list[1], grad_integration_list[2])

    #### test the strength of the representation
    model.eval()
    # compute total number of examples in val set
    total = len(val_loader.dataset) # 2048 for first task
    #print("Attention: evaluating the representation strength for classes: [9, 0]")
    #total = len(forward_transfer_loader.dataset)
    #print("Attention: evaluating the representation strength for ALL classes")
    #total = len(all_classes_loader.dataset)

    # get feature size
    xb, yb, _, next(iter(val_loader))
    #print("Attention: evaluating the representation strength for classes: [9, 0]")
    #xb, yb, _, next(iter(forward_transfer_loader))
    #print("Attention: evaluating the representation strength for ALL classes")
    #xb, yb, _, next(iter(all_classes_loader))
    _, latent = model(xb)
    feat_dim = latent[1].shape[1] # 84

    # preallocate
    all_repr = np.zeros((total, feat_dim), dtype=np.float32)
    all_labels = np.zeros(total, dtype=np.int64)

    print(all_repr.shape)

    ptr = 0
    with torch.no_grad():
        for xb, yb, _ in val_loader:
        #print("Attention: evaluating the representation strength for classes: [9, 0]")
        #for xb, yb, _ in forward_transfer_loader:
        #print("Attention: evaluating the representation strength for ALL classes")
        #for xb, yb, _ in all_classes_loader:
            bs = xb.size(0)
            _, latent = model(xb)
            arr = latent[1].cpu().numpy()

            all_repr[ptr:ptr+bs] = arr
            all_labels[ptr:ptr+bs] = yb.cpu().numpy()
            ptr += bs

    #print("Attention: evaluating the representation strength for classes: [9, 0]")
    #tsne = TSNE(n_components=2, init='pca', random_state=42)
    #latents_2d = tsne.fit_transform(all_repr)  # [N,2]
    #plot_tsne(latents_2d, all_labels, title="MNIST latent space (2D) - Classes [9, 0]")
    #print("Attention: evaluating the representation strength for ALL classes")
    #tsne = TSNE(n_components=2, init='pca', random_state=42)
    #latents_2d = tsne.fit_transform(all_repr)  # [N,2]
    #plot_tsne(latents_2d, all_labels, title="MNIST latent space (2D) - All classes")

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
    #print("Attention: evaluating the representation strength for classes: [9, 0]")
    #seen_counts.append(2)
    #for c in [9, 0]:
    #print("Attention: evaluating the representation strength for ALL classes")
    #seen_counts.append(10)
    #for c in [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]:
        mask = (y_test == c)
        if mask.sum() > 0:
            acc_c = (y_pred[mask] == y_test[mask]).mean()
            print(f"Class {c} accuracy on linear probing: {acc_c:.4f}")
            
            representation_strength[c].append(acc_c)

    plot_class_strength(representation_strength, seen_counts)
    print_representation_strength_table(representation_strength, classifier_acc, len(classifier_acc))

    print("")

    # the current task becomes now the previous task
    previous = task_classes
    #lambda_l1 += 0.5

