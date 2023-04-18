from torch import nn
from utils import he_init


class LeNet5(nn.Module):
    def __init__(self, activate=nn.ReLU):
        super(LeNet5, self).__init__()

        self.activate = activate
        self.reset_activation_func()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(400, 120)
        self.lin2 = nn.Linear(120, 84)
        self.lin3 = nn.Linear(84, 10)

        self.apply(he_init)
        

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
    

    def reset_activation_func(self, **kwargs):
        self.af1 = self.activate(**kwargs)
        self.af2 = self.activate(**kwargs)
        self.af3 = self.activate(**kwargs)
        self.af4 = self.activate(**kwargs)


    def set_activate(self, activate, **kwargs):
        self.activate = activate
        self.reset_activation_func(**kwargs)


    def apply_forward_hook(self, func):
        handles = []
        handles.append(self.af1.register_forward_hook(func))
        handles.append(self.af2.register_forward_hook(func))
        handles.append(self.af3.register_forward_hook(func))
        handles.append(self.af4.register_forward_hook(func))
        return handles