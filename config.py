from src.similarities import cos_sim
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


### MEASURES - for processing script

USER_MEASURES = [
    'sim@10',
    'sim@100',
    'sim@init',
    'sim@final',
    'wc@10',
    'wc@100',
    'wc@init',
    'wc@final',
]

SIM_FUNC_MAP = {
    'cos': cos_sim,
}

SIMIALRITIES = [
    'cos',
]