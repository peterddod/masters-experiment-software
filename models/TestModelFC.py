from collections import OrderedDict
from torch import nn
import torch

class TestModelFC(nn.Module):
    def __init__(self):
        super(TestModelFC, self).__init__()

        self.activate = nn.ReLU

        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(28*28,24)
        self.af1 = self.activate()
        self.lin2 = nn.Linear(24,24)
        self.af2 = self.activate()
        self.lin3 = nn.Linear(24,24)
        self.af3 = self.activate()
        self.lin4 = nn.Linear(24,10)
        

    def forward(self, x):
        x = self.flat(x)
        x = self.lin1(x)
        x = self.af1(x)
        x = self.lin2(x)
        x = self.af2(x)
        x = self.lin3(x)
        x = self.af3(x)
        x = self.lin4(x)
        return x
    

    def reset_activation_func(self):
        self.af1 = self.activate()
        self.af2 = self.activate()
        self.af3 = self.activate()
        self.af4 = self.activate()


    def set_relu(self, set_relu=True):
        if set_relu:
            self.activate = nn.ReLU
        else:
            self.activate = nn.Identity

        self.reset_activation_func()
    

    def apply_hook(self, func):
        self.af1.register_full_backward_hook(func('af1'))
        self.af2.register_full_backward_hook(func('af2'))
        self.af3.register_full_backward_hook(func('af3')) 


    def apply_forward_hook(self, func):
        self.af1.register_forward_hook(func('af1'))
        self.af2.register_forward_hook(func('af2'))
        self.af3.register_forward_hook(func('af3'))