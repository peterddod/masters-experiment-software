from collections import OrderedDict
from torch import nn
import torch

class TestModelLeNet5(nn.Module):
    def __init__(self):
        super(TestModelLeNet5, self).__init__()

        self.activate = nn.ReLU

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.af1 = self.activate()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.af2 = self.activate()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(400, 120)
        self.af3 = self.activate()
        self.lin2 = nn.Linear(120, 84)
        self.af4 = self.activate()
        self.lin3 = nn.Linear(84, 10)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.af1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.af2(x)
        x = self.pool2(x)
        x = self.flat(x)

        x = self.lin1(x)
        x = self.af3(x)
        x = self.lin2(x)
        x = self.af4(x)
        x = self.lin3(x)
        
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
        self.af4.register_full_backward_hook(func('af4'))


    def apply_forward_hook(self, func):
        self.af1.register_forward_hook(func('af1'))
        self.af2.register_forward_hook(func('af2'))
        self.af3.register_forward_hook(func('af3'))
        self.af4.register_forward_hook(func('af4'))