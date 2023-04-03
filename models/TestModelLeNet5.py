from collections import OrderedDict
from torch import nn
import torch

class TestModelLeNet5(nn.Module):
    def __init__(self):
        super(TestModelLeNet5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(400, 120)
        self.lin2 = nn.Linear(120, 84)
        self.lin3 = nn.Linear(84, 10)

        self.activate = nn.ReLU()
        

    def forward(self, x):
        x = self.activate(self.conv1(x))
        x = self.pool1(x)
        x = self.activate(self.conv2(x))
        x = self.pool2(x)
        x = self.flat(x)

        x = self.activate(self.lin1(x))
        x = self.activate(self.lin2(x))
        x = self.lin3(x)
        
        return x
    

    def set_relu(self, set_relu=True):
        if set_relu:
            self.activate = nn.ReLU()
        else:
            self.activate = nn.Identity()
    

    def apply_hook(self, func):
        self.conv1.bias.register_hook(func('conv1'))
        self.conv1.weight.register_hook(func('conv1'))

        self.conv2.bias.register_hook(func('conv2'))
        self.conv2.weight.register_hook(func('conv2'))

        self.lin1.bias.register_hook(func('lin1'))
        self.lin1.weight.register_hook(func('lin1'))

        self.lin2.bias.register_hook(func('lin2'))
        self.lin2.weight.register_hook(func('lin2'))


    def apply_forward_hook(self, func):
        self.conv1.register_forward_hook(func('conv1'))
        self.conv2.register_forward_hook(func('conv2'))
        self.lin1.register_forward_hook(func('lin1'))
        self.lin2.register_forward_hook(func('lin2'))