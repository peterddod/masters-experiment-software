from torch import nn
from utils import he_init


class ExpModelFC(nn.Module):
    def __init__(self):
        super(ExpModelFC, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,24),
            nn.ReLU(),
            nn.Linear(24,24),
            nn.ReLU(),
            nn.Linear(24,24),
            nn.ReLU(),
            nn.Linear(24,10),
        )
        
        self.model.apply(he_init)

    def forward(self, x):
        return self.model(x)