from utils.utils import generate_patterns, plot_curve 
from models.models import MLP
from datasets.datasets import RIDataset

from torch.utils.data import DataLoader
import torch.nn as nn
import torch

seed = 42

lr = 0.1
epochs = 40

AB_inputs, AC_inputs, AB_pairs, AC_pairs = generate_patterns(seed=seed)

ABC_pairs = AB_pairs + AC_pairs 

ABC_list_dataset = RIDataset(ABC_pairs)
#ABListDataset = RIDataset(AB_pairs)
#ACListDataset = RIDataset(AC_pairs)

loader = DataLoader(ABC_list_dataset, batch_size=len(ABC_pairs), shuffle=True)

x, y = next(iter(loader))

model = MLP()

bce_loss = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

loss_list = []
acc_list = []

for epoch in range(epochs):

    model.train()

    for x, y in loader:
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

print(y[0])
print(preds_probs[0])

plot_curve(loss_list)
plot_curve(acc_list)

