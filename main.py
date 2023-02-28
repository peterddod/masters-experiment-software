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
from analysis import PatternCollector


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

    l = PatternCollector(model, datasets.MNIST('./dataset', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])).data.reshape([60000,1,28,28]))

    optimiser = optimiser_map[args.optimiser](model.parameters(), lr=args.theta)

    os.mkdir(f'./results/{args.filename}/')
    file = FileWriter(f'./results/{args.filename}/training_log', ",".join(['epoch','step','loss','unique_aps','mean_ap_sim','std_ap_sim']))

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./dataset', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=args.batchsize, 
        shuffle=True,
    )

    total = 0
    # correct = 0
    # train_loss = 0
    loss = loss_map[args.loss]()

    for epoch in range(1, args.epochs+1):
        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
            optimiser.zero_grad()
            output = model(data)
            output = loss(output, target)
            output.backward()
            optimiser.step()

            total += data.size(0)
            # correct += (output.argmax(dim=1) == target).float().sum()
            # train_loss += loss.item() * data.size(0)

            if batch_idx % 10 == 0:      
                l.update()

                file(",".join([
                    str(epoch),
                    str(batch_idx+1),
                    str(output.item()),
                    str(l.getNumberOfUniquePatterns()),
                    str(l.getMeanCosineOfPatternsToLastStep()),
                    str(l.getStdCosineOfPatternsToLastStep()),
                ]))

                print(f'Epoch {epoch}, Batch: {batch_idx}, Loss: {output.item()}')
                print(l.getMeanCosineOfPatternsToLastStep())
                print(l.getNumberOfUniquePatterns())  


    t_end = time()

    file(f'time: {t_end-t_start}') #