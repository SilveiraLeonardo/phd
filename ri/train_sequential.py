from utils.utils import generate_patterns, plot_curve, plot_accuracies 
from models.models import MLP
from datasets.datasets import RIDataset

from torch.utils.data import DataLoader
import torch.nn as nn
import torch

seed = 42

lr = 0.1
epochs = 20

AB_inputs, AC_inputs, AB_pairs, AC_pairs = generate_patterns(seed=seed)

ABC_pairs = AB_pairs + AC_pairs 

#ABC_list_dataset = RIDataset(ABC_pairs)
AB_list_dataset = RIDataset(AB_pairs)
AC_list_dataset = RIDataset(AC_pairs)

AB_loader = DataLoader(AB_list_dataset, batch_size=len(AB_pairs), shuffle=True)
AC_loader = DataLoader(AC_list_dataset, batch_size=len(AC_pairs), shuffle=True)

model = MLP()

bce_loss = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

loss_list = []
acc_list = []

print('starting AB training...')

for epoch in range(epochs):

    model.train()

    for x, y in AB_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = bce_loss(logits, y)
        loss.backward()
        optimizer.step()

        preds_probs = torch.sigmoid(logits)
        preds = (preds_probs > 0.5).float()
        all_bits_correct = (preds == y).all(dim=1).float().mean().item()

    print(f'Epoch {epoch}, loss {loss:.4f}, acc {all_bits_correct:.4f}')
    loss_list.append(loss.item())
    acc_list.append(all_bits_correct)

#plot_curve(loss_list)
#plot_curve(acc_list)

loss_list = []

acc_list_ac = []
acc_list_ab = []

print('starting AC training...')

epochs *= 3

for epoch in range(epochs):

    model.train()

    for x, y in AC_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = bce_loss(logits, y)
        loss.backward()
        optimizer.step()

        preds_probs = torch.sigmoid(logits)
        preds = (preds_probs > 0.5).float()
        correct_ac = (preds == y).all(dim=1).float().mean().item()
    
    with torch.no_grad():

        for x, y in AB_loader:
            logits = model(x)
            preds_probs = torch.sigmoid(logits)
            preds = (preds_probs > 0.5).float()
            correct_ab = (preds == y).all(dim=1).float().mean().item()

    print(f'Epoch {epoch}, loss {loss:.4f}, acc AC {correct_ac:.4f}, acc AB {correct_ab:.4f}')
    loss_list.append(loss.item())
    acc_list_ac.append(correct_ac)
    acc_list_ab.append(correct_ab)

#plot_accuracies(acc_list_ac, acc_list_ab)

AB_loader_t = DataLoader(AB_list_dataset, batch_size=1, shuffle=False)
AC_loader_t = DataLoader(AC_list_dataset, batch_size=1, shuffle=False)

AB_example_x, AB_example_y = next(iter(AB_loader_t))
AC_example_x, AC_example_y = next(iter(AC_loader_t))

logits = model(AB_example_x)
preds_probs = torch.sigmoid(logits)
preds_AB = (preds_probs > 0.5).float()[0].tolist()

logits = model(AC_example_x)
preds_probs = torch.sigmoid(logits)
preds_AC = (preds_probs > 0.5).float()[0].tolist()

print("AC input:")
print(AC_example_x[0].tolist())
print("AC output:")
print(preds_AC)
print("AC target:")
print(AC_example_y[0].tolist())
print(" ")
print("AB input:")
print(AB_example_x[0].tolist())
print("AB output:")
print(preds_AB)
print("AB target:")
print(AB_example_y[0].tolist())

