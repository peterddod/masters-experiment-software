from models import ResNet9, LeNet5, SimpleMLP
from torch import nn
from torch import optim


### PARAMATER MAPS

MODELS = {
    'exp_fc': SimpleMLP,
    'exp_lenet': LeNet5,
    'exp_resnet': ResNet9,
}

OPTIMISERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
}

LOSSES = {
    'nll': nn.NLLLoss,
    'mse': nn.MSELoss,
    'crossentropy': nn.CrossEntropyLoss,
}


### PATH MAP

PATHS = {
    'results': {
        'main': './results/main/',
        'process': './results/process/',
        'test': './results/test/',
    }
}


