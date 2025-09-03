import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size=20, hidden_size=50, output_size=10):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        self.h1 = F.relu(self.linear1(x))
        logits = self.linear2(self.h1)

        return logits
