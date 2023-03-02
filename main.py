"""
Script for running experiments.

Part of MSci Project for Peter Dodd @ University of Glasgow.
"""
import torch
import os
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np  
from time import time
from datetime import datetime
from models import SmallFC
from utils import *
import argparse
import torch.multiprocessing as mp
from train import *
from analysis import analyse_model, extract
import warnings
import json


warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()


### SOFTWARE PARAMETERS

parser.add_argument("-f", "--filename", default=f'{datetime.now().strftime("%d.%m.%Y %H.%M.%S")}', help="Name of output folder")
parser.add_argument("-s", "--seed", default=1, help="Seed to run experiment on", type=int)


### TRAINING PARAMETERS

parser.add_argument("-e", "--epochs", default=2, help="Number of training epochs", type=int)
parser.add_argument("-b", "--batchsize", default=64, help="Size of each training batch", type=int)
parser.add_argument("-m", "--model", default='small_fc', help="The model to run training for")
parser.add_argument("-o", "--optimiser", default='sgd', help="Optimiser for training")
parser.add_argument("-l", "--loss", default='nll', help="Loss function for training")
parser.add_argument("-t", "--theta", default=0.01, help="Learning rate", type=int)


### PARAMATER MAPS

model_map = {
    'small_fc': SmallFC,
}

optimiser_map = {
    'sgd': optim.SGD
}

loss_map = {
    'nll': nn.NLLLoss,
    'mse': nn.MSELoss,
}


### SCRIPT

if __name__ == '__main__':
    args = parser.parse_args()

    t_start = time()

    resetseed(args.seed)

    model = model_map[args.model]()
    optimiser = optimiser_map[args.optimiser](model.parameters(), lr=args.theta)

    pattern_data = datasets.MNIST('./dataset', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])).data.reshape([60000,1,28,28]).to(torch.float32)

    os.mkdir(f'./results/{args.filename}/')
    file = FileWriter(f'./results/{args.filename}/training.log', ",".join(['epoch','step','train_loss','test_accuracy']))
    pattern_file = ByteWriter(f'./results/{args.filename}/patterns.bytes')

    activation_idxs, skip_idxs, layer_output_sizes = analyse_model(model, pattern_data, pattern_file)

    info_dictionary = {
        'script_paramaters': {
            **dict(args._get_kwargs()),
        },
        'layer_output_sizes': layer_output_sizes,
    }

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./dataset', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=args.batchsize, 
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./dataset', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=256, 
        shuffle=False,
    )

    total = 0
    loss = loss_map[args.loss]()

    for epoch in range(1, args.epochs+1):
        for batch_idx, (data, target) in enumerate(train_loader):
            p_train_epoch = mp.Process(target=run_epoch, args=(
                model, optimiser, loss, 
                data, test_loader,
                target, epoch,
                batch_idx,
                file
            )) 
            p_pattern_analysis = mp.Process(target=extract, args=(
                model.state_dict(),
                model_map[args.model],
                skip_idxs,
                activation_idxs,
                pattern_data,
                pattern_file,
            )) 

            p_train_epoch.start()
            p_pattern_analysis.start()

            p_train_epoch.join()
            p_pattern_analysis.join()

    t_end = time()

    info_dictionary['execution_time'] = t_end-t_start

    with open(f'./results/{args.filename}/experiment.info', 'w') as f:
        json.dump(info_dictionary, f, indent = 4) 
        f.close()
