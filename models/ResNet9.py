from torch import nn
from utils import he_init
from .modules import ResidualBlock


class ResNet9(nn.Module):
    def __init__(self, activate=nn.ReLU):
        super(ResNet9, self).__init__()

        self.activate = activate
        self.reset_activation_func()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64, momentum=0.9)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=128, momentum=0.9)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res1 = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=256, momentum=0.9)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=256, momentum=0.9)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res2 = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flat = nn.Flatten()
        self.lin = nn.Linear(in_features=1024, out_features=10, bias=True)

        self.apply(he_init)
        self.res1.apply(he_init)
        self.res2.apply(he_init)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.af1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.af2(x)
        x = self.pool1(x)
        x = self.res1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.af3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.af4(x)
        x = self.pool3(x)
        x = self.res2(x)
        x = self.pool4(x)

        x = self.flat(x)
        x = self.lin(x)
        
        return x
    

    def reset_activation_func(self, **kwargs):
        self.af1 = self.activate(**kwargs)
        self.af2 = self.activate(**kwargs)
        self.af3 = self.activate(**kwargs)
        self.af4 = self.activate(**kwargs)


    def set_activate(self, activate, **kwargs):
        self.activate = activate
        self.reset_activation_func(**kwargs)
        self.res1.set_activate(activate, **kwargs)
        self.res2.set_activate(activate, **kwargs)


    def apply_forward_hook(self, func):
        handles = []
        handles.append(self.af1.register_forward_hook(func))
        handles.append(self.af2.register_forward_hook(func))
        handles = [*handles,*self.res1.apply_forward_hook(func)]
        handles.append(self.af3.register_forward_hook(func))
        handles.append(self.af4.register_forward_hook(func))
        handles = [*handles,*self.res2.apply_forward_hook(func)]
        return handles