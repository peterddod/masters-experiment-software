from collections import OrderedDict
from torch import nn
import torch

class TestModelFC(nn.Module):
    def __init__(self):
        super(TestModelFC, self).__init__()

        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(28*28,24)
        self.lin2 = nn.Linear(24,24)
        self.lin3 = nn.Linear(24,24)
        self.lin4 = nn.Linear(24,10)

        self.activate = nn.ReLU()
        

    def forward(self, x):
        x = self.flat(x)
        x = self.activate(self.lin1(x))
        x = self.activate(self.lin2(x))
        x = self.activate(self.lin3(x))
        x = self.lin4(x)
        return x
    

    def set_relu(self, set_relu=True):
        if set_relu:
            self.activate = nn.ReLU()
        else:
            self.activate = nn.Identity()
    

    def apply_hook(self, func):
        self.lin1.bias.register_hook(func('lin1'))
        self.lin2.bias.register_hook(func('lin2'))
        self.lin3.bias.register_hook(func('lin3'))


    def apply_forward_hook(self, func):
        self.lin1.register_forward_hook(func('lin1'))
        self.lin2.register_forward_hook(func('lin2'))
        self.lin3.register_forward_hook(func('lin3'))