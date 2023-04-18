"""
Script to process raw byte data produced by main.py.

Part of MSci Project for Peter Dodd @ University of Glasgow.
"""
import argparse
import json

import torch
from src import MeasureCollector
from src.activation import cache_pattern_matrix, delete_unused, get_filenames_to_process, get_prev, process, process_statistics
from utils import get_train_loaders
from config import *
import os


parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", default='test', help="Name of input folder from /results")
parser.add_argument("-f", "--filename", default=None, help="Name of output file")  # TODO: get default value from experiment.info
parser.add_argument("-ds", "--dataset", default='mnist', help="Dataset to train with")
parser.add_argument("-b", "--batchsize", default=10, help="Number of patterns to cache and process", type=int)

parser.add_argument("-u", "--uniquepatterns", action='store_false', help="Controls whether or not to record number of unique patterns measure (turn this off in large networks)")
parser.add_argument("-s", "--samplerate", default=10, help="Gap between gradient updates for comparisons", type=int)
parser.add_argument("-d", "--device", default='cpu', help="Device for processing")
parser.add_argument("-S", "--seed", default=None, help="Seed for subsampling similarity dataset", type=int)
parser.add_argument("-n", default=None, help="Number of samples to use for activation comparison", type=int)


if __name__ == '__main__':
    args = parser.parse_args()

    filename = args.filename

    results_path = f'{PATHS["results"]["process"]}{filename}/'
    cache_path = f'{results_path}.cache/'
    import_path = f'{PATHS["results"]["main"]}{args.input}/'
    snapshots_path = f'{import_path}snapshots/'

    # import experiment info
    with open(f'{import_path}info.json', 'r') as f:
        experiment_info = json.load(f)
        f.close()

    os.mkdir(results_path)
    os.mkdir(cache_path)

    base_measures = [
        'epoch',
        'batch_idx',
        'test_acc',
    ]

    measures = [*base_measures, *USER_MEASURES]

    measure_collector = MeasureCollector(
        measures,
        SIMIALRITIES,
        f'{results_path}processed.csv',
    )
    
    # 1. find range of updates required to process one set of stats 
    model_cls = MODELS[experiment_info['script_parameters']['model']]

    # pattern_data = get_pattern_data(args.dataset)
    test_loader = get_train_loaders(args.dataset, seed=args.seed, n=args.n)[1]

    comparisons = [1,10]  # how many update steps between comparisons
    batch_idx = 0

    # cache patterns for absolute similarity checks
    cache_pattern_matrix('init', test_loader, model_cls, snapshots_path, cache_path, args.device)  # pattern matrix at initialisation
    cache_pattern_matrix('final', test_loader, model_cls, snapshots_path, cache_path, args.device)  # final pattern matrix

    updates_in_epoch = experiment_info['dataset_size'] // experiment_info['script_parameters']['batchsize']
    total_number_of_updates = updates_in_epoch * experiment_info['script_parameters']['epochs']

    # 3. extract patterns and store in .pt files from that amount of .pt snapshots
    storage_window_filenames = get_filenames_to_process(
        args.samplerate, 
        comparisons, 
        args.batchsize, 
        batch_idx, 
        updates_in_epoch, 
        total_number_of_updates
    ) 

    processing_window_filenames = storage_window_filenames

    while storage_window_filenames != None:
        func = lambda filename: cache_pattern_matrix(
            filename,
            test_loader,
            model_cls, 
            snapshots_path,
            cache_path,
            args.device
        )
        
        for x in storage_window_filenames: func(x)

        for filename in processing_window_filenames:
            activation_matrix = torch.load(f'{cache_path}{filename}.pt').detach()

            model = model_cls()
            model.load_state_dict(torch.load(f'{snapshots_path}{filename}.pt', map_location=torch.device('cpu')))

            current_measure = measure_collector.next()

            while current_measure != None:                
                result = process(
                    current_measure,
                    filename = filename,
                    test_loader = test_loader,
                    model_cls = model_cls,
                    snapshots_path = snapshots_path,
                    cache_path = cache_path,
                    samplerate = args.samplerate,
                    updates_in_epoch = updates_in_epoch,
                    device = args.device,
                    activation_matrix = activation_matrix,
                    model = model,
                )

                measure_collector.add(result)
                current_measure = measure_collector.next()
            
            measure_collector.write()

        # func = lambda filename: process_statistics(
        #     filename,
        #     output,
        #     test_loader,
        #     model_cls,
        #     get_prev(filename, comparisons[0], args.samplerate, updates_in_epoch),
        #     get_prev(filename, comparisons[1], args.samplerate, updates_in_epoch),
        #     args.uniquepatterns,
        #     snapshots_path,
        #     cache_path,
        #     args.device
        # )
        
        # [func(x) for x in processing_window_filenames]

        batch_idx += 1

        # 5. delete a windows worth of .bin files from the start that aren't used anymore and process a windows worth more updates
        storage_window_filenames_old = storage_window_filenames
        storage_window_filenames = get_filenames_to_process(
            args.samplerate, 
            comparisons, 
            args.batchsize, 
            batch_idx, 
            updates_in_epoch, 
            total_number_of_updates
        )  

        if (storage_window_filenames != None):
            delete_unused(storage_window_filenames_old, storage_window_filenames, cache_path)  # delete files that exist in old that aren't in new
            processing_window_filenames = storage_window_filenames[args.batchsize:]

