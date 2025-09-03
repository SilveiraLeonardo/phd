import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

import numpy as np
import itertools

def generate_patterns(seed=1234):
    rng = np.random.default_rng(seed)

    # all 1024 possible 10-bit patterns  
    all_patterns = np.array(list(itertools.product([0,1], repeat=10)), dtype=int)

    # choose 26 distinct indices (8 A + 2 contexts + 8 B + 8 C)
    idx = rng.choice(len(all_patterns), size=26, replace=False)

    # assign
    A_patterns      = all_patterns[idx[0:8]]    # A₁…A₈
    ctx_B_pattern   = all_patterns[idx[8]]      # single B-context
    ctx_C_pattern   = all_patterns[idx[9]]      # single C-context
    B_response      = all_patterns[idx[10:18]]  # B₁…B₈
    C_response      = all_patterns[idx[18:26]]  # C₁…C₈

    # Now form the 8 AB pairs and the 8 AC pairs
    AB_inputs  = np.hstack([A_patterns, np.tile(ctx_B_pattern, (8,1))])
    AC_inputs  = np.hstack([A_patterns, np.tile(ctx_C_pattern, (8,1))])
    AB_pairs   = list(zip(AB_inputs, B_response))
    AC_pairs   = list(zip(AC_inputs, C_response))

    return AB_inputs, AC_inputs, AB_pairs, AC_pairs 

def plot_curve(line, xlabel='Epoch', ylabel='Loss'): 

    plt.figure(figsize=(6,4))
    plt.plot(range(len(line)), line,  marker='o')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(line)))           # show each batch on the x-axis
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()

    plt.show()

def plot_accuracies(line1, line2, line1_title='Current Task', line2_title='Previous Task',
                    xlabel='Epoch #', ylabel='Accuracy', 
                    title='Accuracy: Current vs. Previous Task'):

    batches = list(range(1, len(line1) + 1))

    plt.figure(figsize=(6,4))
    plt.plot(batches, line1,  marker='o', label=line1_title)
    plt.plot(batches, line2,  marker='s', label=line2_title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0,1)                 # accuracy between 0 and 1
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_tsne(latents_2d, all_labels, title="Latent space (2D)"):

    plt.figure(figsize=(8,8))
    sc = plt.scatter(latents_2d[:,0],
                     latents_2d[:,1],
                     c=all_labels,
                     cmap='tab10',
                     s=50,
                     alpha=0.7)
    plt.legend(*sc.legend_elements(), title="classes")
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.grid(False)
    plt.show()




