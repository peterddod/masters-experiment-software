
"""
Script for testing freezing structure.

Part of MSci Project for Peter Dodd @ University of Glasgow.
"""
import os
from time import time
from datetime import datetime
from utils import *
import argparse
from src.train import *
import warnings
import json
from config import *

from models import TestModelFC, TestModelLeNet5, TestModelAFFC, ExpModelFC, FreezeNet
from models.modules import Activator


warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()


### SOFTWARE PARAMETERS

parser.add_argument("-f", "--filename", default=f'{datetime.now().strftime("%d.%m.%Y %H.%M.%S")}', help="Name of output folder")
parser.add_argument("-s", "--seed", default=1, help="Seed to run experiment on", type=int)
parser.add_argument("-sr", "--samplerate", default=10, help="Seed to run experiment on", type=int)
parser.add_argument("-ds", "--dataset", default='mnist', help="Dataset to train with")


### TRAINING PARAMETERS

parser.add_argument("-e", "--epochs", default=2, help="Number of training epochs", type=int)
parser.add_argument("-b", "--batchsize", default=64, help="Size of each training batch", type=int)
parser.add_argument("-m", "--model", default='small_fc', help="The model to run training for")
parser.add_argument("-o", "--optimiser", default='sgd', help="Optimiser for training")
parser.add_argument("-l", "--loss", default='nll', help="Loss function for training")
parser.add_argument("-t", "--theta", default=0.01, help="Learning rate", type=float)
parser.add_argument("-w", "--weight_decay", default=0, help="Weight decay", type=float)
parser.add_argument("-fp", "--freezepoint", default=2, help="Epoch at which to freeze structure", type=int)
parser.add_argument("-d", "--device", default='cpu', help="Device for training")


### SCRIPT

if __name__ == '__main__':
    args = parser.parse_args()

    t_start = time()

    resetseed(args.seed)

    model = TestModelLeNet5()
    m = TestModelLeNet5()
    # model.to(args.device)

    optimiser = optimiser_map[args.optimiser](list(model.parameters()) + list(m.parameters()), lr=args.theta)

    os.mkdir(f'./test_results/{args.filename}/')

    file = FileWriter(f'./test_results/{args.filename}/log.csv', ",".join(['epoch','step','train_loss','test_accuracy','train_accuracy']))

    train_loader, test_loader = get_train_loaders(args.dataset, args.batchsize)

    info_dictionary = {
        'script_parameters': {
            **dict(args._get_kwargs()),
        },
        'dataset_size': len(train_loader.dataset)
    }

    total = 0
    loss = loss_map[args.loss]()

    for epoch in range(1, args.epochs+1):
        if epoch == args.freezepoint:
            m.load_state_dict(model.state_dict())
            model = FreezeNet(model, m)
            model.model.set_activate(Activator, get_activation_matrix=model.get_activation_matrix)
            optimiser = optimiser_map[args.optimiser](model.parameters(), lr=args.theta)

        for batch_idx, (data, target) in enumerate(train_loader):
            run_step_w_acc(
                model, optimiser, loss, 
                data, test_loader, train_loader,
                target, epoch,
                batch_idx,
                file,
                args.device
            )

    t_end = time()

    info_dictionary['execution_time'] = t_end-t_start

    with open(f'./test_results/{args.filename}/info.json', 'w') as f:
        json.dump(info_dictionary, f, indent = 4) 
        f.close()
