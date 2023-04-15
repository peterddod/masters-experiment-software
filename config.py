from models import ResNet9, LeNet5, SimpleMLP
from torch import nn
import torch.optim as optim


### PARAMATER MAPS

model_map = {
    'exp_fc': SimpleMLP,
    'exp_lenet': LeNet5,
    'exp_resnet': ResNet9,
}

optimiser_map = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
}

loss_map = {
    'nll': nn.NLLLoss,
    'mse': nn.MSELoss,
    'crossentropy': nn.CrossEntropyLoss,
}
