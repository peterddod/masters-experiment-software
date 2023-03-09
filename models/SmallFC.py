from torch import nn


class SmallFC(nn.Module):
    def __init__(self, hidden_units=24):
        super(SmallFC, self).__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden_units),
            nn.ReLU(),
            nn.LazyLinear(hidden_units),
            nn.ReLU(),
            nn.LazyLinear(hidden_units),
            nn.ReLU(),
            nn.LazyLinear(10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.model(x)