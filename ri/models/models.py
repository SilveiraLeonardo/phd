import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size=20, hidden_size=50, output_size=10, p=0.1):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(p)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(p)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        self.h1 = F.relu(self.linear1(x))
        self.h2 = F.relu(self.linear2(self.dropout1(self.h1)))
        logits = self.linear3(self.dropout2(self.h2))

        return logits

class MLPSparse(nn.Module):
    def __init__(self, input_size=20, hidden_size=50, output_size=10):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        h1 = F.relu(self.linear1(x))
        h2 = F.relu(self.linear2(h1))
        logits = self.linear3(h2)

        return logits, (h1, h2)

