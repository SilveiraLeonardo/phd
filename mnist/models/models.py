import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5) # in: 28x28, out: 24x24
        self.pool = nn.MaxPool2d(2, 2) # out: 12x12
        self.conv2 = nn.Conv2d(6, 16, 5) # out: 8x8
        # pool, out: 4x4
        self.fc1 = nn.Linear(16*4*4, 120) # 256 to 120
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        z = F.relu(self.fc2(x))
        logits = self.fc3(z)
        return logits, z

