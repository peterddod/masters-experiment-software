import torch.nn as nn
from utils import *


class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, activate=nn.ReLU):
        super(ResidualBlock, self).__init__()
        self.activate = activate
        self.reset_activation_func()

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


    def forward(self, x):
        residual = x

        out = self.af1(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.af2(out)

        return out + residual
    

    def reset_activation_func(self, **kwargs):
        self.af1 = self.activate(**kwargs)
        self.af2 = self.activate(**kwargs)


    def set_activate(self, activate, **kwargs):
        self.activate = activate
        self.reset_activation_func(**kwargs)
    
    
    def apply_forward_hook(self, func):
        handles = []
        handles.append(self.af1.register_forward_hook(func))
        handles.append(self.af2.register_forward_hook(func))
        return handles