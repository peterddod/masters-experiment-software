"""
Code adapted from: https://github.com/matthias-wright/cifar10-resnet/blob/master/model.py
"""

import torch.nn as nn
from collections import OrderedDict


class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)

        return out + residual
    
    def apply_activation_hook(self, func):
        self.relu.register_forward_hook(func)


class ExpResNet9(nn.Module):
    """
    A Residual network.
    """
    def __init__(self):
        super(ExpResNet9, self).__init__()

        self.conv = nn.Sequential(OrderedDict([
            ('conv1',nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn1',nn.BatchNorm2d(num_features=64, momentum=0.9)),
            ('relu1',nn.ReLU()),
            ('conv2',nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2',nn.BatchNorm2d(num_features=128, momentum=0.9)),
            ('relu2',nn.ReLU()),
            ('pool1',nn.MaxPool2d(kernel_size=2, stride=2)),
            ('res1',ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)),
            ('conv3',nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn3',nn.BatchNorm2d(num_features=256, momentum=0.9)),
            ('relu3',nn.ReLU()),
            ('pool2',nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv4',nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn4',nn.BatchNorm2d(num_features=256, momentum=0.9)),
            ('relu4',nn.ReLU()),
            ('pool3',nn.MaxPool2d(kernel_size=2, stride=2)),
            ('res2',ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)),
            ('pool4',nn.MaxPool2d(kernel_size=2, stride=2)),
            ('flat',nn.Flatten()),
            ('lin',nn.Linear(in_features=1024, out_features=10, bias=True))
        ]))

    def forward(self, x):
        return self.conv(x)
    
    def apply_activation_hook(self, func):
        self.conv.relu1.register_forward_hook(func)
        self.conv.relu2.register_forward_hook(func)
        self.conv.res1.apply_activation_hook(func)
        self.conv.relu3.register_forward_hook(func)
        self.conv.relu4.register_forward_hook(func)
        self.conv.res2.apply_activation_hook(func)