from collections import OrderedDict
from torch import nn
from utils import he_init


class ExpLeNet5(nn.Module):
    def __init__(self):
        super(ExpLeNet5, self).__init__()

        self.model = nn.Sequential(OrderedDict([
            ('conv1',nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)),
            ('relu1',nn.ReLU()),
            ('pool1',nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2',nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)),
            ('relu2',nn.ReLU()),
            ('pool2',nn.MaxPool2d(kernel_size=2, stride=2)),
            ('falt',nn.Flatten()),
            ('lin1',nn.Linear(400, 120)),
            ('relu3',nn.ReLU()),
            ('lin2',nn.Linear(120, 84)),
            ('relu4',nn.ReLU()),
            ('lin3',nn.Linear(84, 10))
        ]))
        
        self.model.apply(he_init)

    def forward(self, x):
        return self.model(x)
    
    def apply_activation_hook(self, func):
        self.model.relu1.register_forward_hook(func)
        self.model.relu2.register_forward_hook(func)
        self.model.relu3.register_forward_hook(func)
        self.model.relu4.register_forward_hook(func)