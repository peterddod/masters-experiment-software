
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

from models import FreezeNet


warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()


### SOFTWARE PARAMETERS

parser.add_argument("-f", "--filename", default=f'{datetime.now().strftime("%d.%m.%Y %H.%M.%S")}', help="Name of output folder")
parser.add_argument("-s", "--seed", default=1, help="Seed to run experiment on", type=int)
parser.add_argument("-sr", "--samplerate", default=10, help="Seed to run experiment on", type=int)


### TRAINING PARAMETERS

parser.add_argument("-e", "--epochs", default=2, help="Number of training epochs", type=int)
parser.add_argument("-b", "--batchsize", default=64, help="Size of each training batch", type=int)
parser.add_argument("-m", "--model", default='exp_fc', help="The model to run training for")
parser.add_argument("-o", "--optimiser", default='sgd', help="Optimiser for training")
parser.add_argument("-l", "--loss", default='nll', help="Loss function for training")
parser.add_argument("-t", "--theta", default=0.01, help="Learning rate", type=float)
parser.add_argument("-w", "--weight_decay", default=0, help="Weight decay", type=float)
parser.add_argument("-fp", "--freezepoint", default=2, help="Epoch at which to freeze structure", type=int)
parser.add_argument("-d", "--device", default='cpu', help="Device for training")
parser.add_argument("-ds", "--dataset", default='mnist', help="Dataset to train with")
parser.add_argument("-i", "--input", default=None, help="Specific folder from main.py training to take pattern selector from")


### SCRIPT

if __name__ == '__main__':
    args = parser.parse_args()
    torch.autograd.set_detect_anomaly(True)

    t_start = time()

    resetseed(args.seed)

    model = MODELS[args.model]()
    model.to(args.device)

    optimiser = OPTIMISERS[args.optimiser](model.parameters(), lr=args.theta)

    start_epoch = 1

    if args.input != None:
        import_path = f'{PATHS["results"]["main"]}{args.input}/'
        snapshots_path = f'{import_path}snapshots/'
        model.load_state_dict(torch.load(f'{snapshots_path}{args.freezepoint}.pt', map_location=torch.device(args.device)))

        optimiser_params = torch.load(f'{snapshots_path}{args.freezepoint}_optim.pt', map_location=torch.device(args.device))
        optimiser = OPTIMISERS[args.optimiser](model.parameters(), **optimiser_params)

        start_epoch = args.freezepoint


    results_path = f'{PATHS["results"]["test"]}{args.filename}/'

    os.mkdir(results_path)

    file = FileWriter(f'{results_path}log.csv', ",".join(['epoch','step','train_loss','test_accuracy','train_accuracy']))

    resetseed(args.seed)
    train_loader, test_loader = get_train_loaders(args.dataset, args.batchsize, seed=args.seed)

    info_dictionary = {
        'script_parameters': {
            **dict(args._get_kwargs()),
        },
        'dataset_size': len(train_loader.dataset)
    }

    total = 0
    resetseed(args.seed)
    loss = LOSSES[args.loss]()

    for epoch in range(start_epoch, args.epochs+1):
        if epoch == args.freezepoint:
            model_f = FreezeNet(MODELS[args.model], pattern_selector=model.state_dict(), seed=args.seed)
            model_f.freeze()

            # this small block means the optimiser for the freeze network has the same adapted parameters as the original
            in_params = optimiser.param_groups[0]
            if 'params' in in_params: del in_params['params']
            optimiser = OPTIMISERS[args.optimiser](model_f.parameters(), **in_params)
            
            model_f.to(args.device)
            model = model_f

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

    with open(f'{results_path}info.json', 'w') as f:
        json.dump(info_dictionary, f, indent = 4) 
        f.close()
