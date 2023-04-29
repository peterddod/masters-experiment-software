"""
Script for running experiments.

Part of MSci Project for Peter Dodd @ University of Glasgow.
"""
import os
from time import time
from datetime import datetime
from src.analysis import get_layer_start_indexes
from utils import *
import argparse
from src.train import *
import warnings
import json
from config import *


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
parser.add_argument("-d", "--device", default='cpu', help="Device for training")


### SCRIPT

if __name__ == '__main__':
    args = parser.parse_args()
    torch.autograd.set_detect_anomaly(True)

    t_start = time()

    resetseed(args.seed)

    model = MODELS[args.model]()
    model.to(args.device)

    optimiser = OPTIMISERS[args.optimiser](model.parameters(), lr=args.theta)

    results_path = f"{PATHS['results']['main']}{args.filename}/"
    snapshot_path = f'{results_path}snapshots/'

    os.mkdir(results_path)
    os.mkdir(snapshot_path)

    file = FileWriter(f'{results_path}log.csv', ",".join(['epoch','step','train_loss','test_accuracy']))

    resetseed(args.seed)
    train_loader, test_loader = get_train_loaders(args.dataset, args.batchsize, seed=args.seed)

    info_dictionary = {
        'script_parameters': {
            **dict(args._get_kwargs()),
        },
        'dataset_size': len(train_loader.dataset),
        'layer_indexes': get_layer_start_indexes(model,train_loader,args.device)
    }

    total = 0
    resetseed(args.seed)
    loss = LOSSES[args.loss]()

    save_state_dict(model.state_dict(), f'{snapshot_path}init.pt')

    updates_in_epoch = info_dictionary['dataset_size'] // args.batchsize

    for epoch in range(1, args.epochs+1):
        save_state_dict(model.state_dict(), f'{snapshot_path}{epoch}.pt')
        for batch_idx, (data, target) in enumerate(train_loader):
            take_snapshot = (batch_idx + (epoch-1)*updates_in_epoch) % args.samplerate == 0

            run_epoch(
                model, optimiser, loss, 
                data, test_loader,
                target, epoch,
                batch_idx,
                file,
                args.device
            )

            if take_snapshot:
                save_state_dict(model.state_dict(), f'{snapshot_path}{epoch}_{batch_idx}.pt')

    save_state_dict(model.state_dict(), f'{snapshot_path}final.pt')

    t_end = time()

    info_dictionary['execution_time'] = t_end-t_start

    with open(f'{results_path}info.json', 'w') as f:
        json.dump(info_dictionary, f, indent = 4) 
        f.close()
