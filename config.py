from models import ExpLeNet5, ExpModelFC, SmallFC
from torch import nn
import torch.optim as optim


### PARAMATER MAPS

model_map = {
    'small_fc': SmallFC,
    'exp_fc': ExpModelFC,
    'exp_lenet': ExpLeNet5,
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
