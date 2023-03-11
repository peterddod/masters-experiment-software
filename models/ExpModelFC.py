from collections import OrderedDict
from torch import nn
from utils import he_init


class ExpModelFC(nn.Module):
    def __init__(self):
        super(ExpModelFC, self).__init__()

        self.model = nn.Sequential(OrderedDict([
            ('flat',nn.Flatten()),
            ('lin1',nn.Linear(28*28,24)),
            ('relu1',nn.ReLU()),
            ('lin2',nn.Linear(24,24)),
            ('relu2',nn.ReLU()),
            ('lin3',nn.Linear(24,24)),
            ('relu3',nn.ReLU()),
            ('lin4',nn.Linear(24,10)),
        ]))
        
        self.model.apply(he_init)

    def forward(self, x):
        return self.model(x)
    
    def apply_activation_hook(self, func):
        self.model.relu1.register_forward_hook(func)
        self.model.relu2.register_forward_hook(func)
        self.model.relu3.register_forward_hook(func)