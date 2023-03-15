"""
Script to process raw byte data produced by main.py.

Part of MSci Project for Peter Dodd @ University of Glasgow.
"""
import argparse
from datetime import datetime
from time import time
import json
import numpy as np
from src.activation import cache_pattern_matrix, delete_unused, get_filenames_to_process, get_prev, make_activation_matrix, process_statistics
from utils import FileWriter, ceil_to_factor
import torch.multiprocessing as mp
from config import *
from torchvision import datasets, transforms
import torch
import os
from functools import partial


parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", default='test', help="Name of input folder from /results")
parser.add_argument("-f", "--filename", default=None, help="Name of output file")  # TODO: get default value from experiment.info

parser.add_argument("-s", "--samplerate", default=10, help="Gap between gradient updates for comparisons", type=int)


if __name__ == '__main__':
    args = parser.parse_args()

    # read experiment.info file to get activation layer sizes
    with open(f'./results/{args.input}/info.json', 'r') as f:
        experiment_info = json.load(f)
        f.close()

    filename = args.filename  # TODO: add folder creation for if /processed doesn't exist

    os.mkdir(f'./processed/{filename}/')
    os.mkdir(f'./processed/{filename}/.cache/')
    
    # create output file
    output = FileWriter(f'./processed/{filename}/processed.csv', ",".join([
        'epoch',
        'batch_idx',
        'test_acc',
        'unique_ap',
        'mean_ap_sim@10','std_ap_sim@10',
        'mean_ap_sim@100','std_ap_sim@100',
        'mean_ap_sim@init','std_ap_sim@init',
        'mean_ap_sim@final','std_ap_sim@final',
        'mean_wc@10','std_wc@10',
        'mean_wc@100','std_wc@100',
        'mean_wc@init','std_wc@init',
        'mean_wc@final','std_wc@final'
        ]))
    
    
    # 1. find range of updates required to process one set of stats 
    model_cls = model_map[experiment_info['script_parameters']['model']]

    pattern_data = datasets.MNIST('./dataset', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])).data.reshape([60000,1,28,28]).to(torch.float32)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./dataset', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=256, 
        shuffle=False,
    )

    comparisons = [1,10]  # how many update steps between comparisons
    batch_size = 10
    batch_idx = 0

    # cache patterns for absolute similarity checks
    cache_pattern_matrix('init', args.input, pattern_data, model_cls)  # pattern matrix at initialisation
    cache_pattern_matrix('final', args.input, pattern_data, model_cls)  # final pattern matrix

    updates_in_epoch = experiment_info['dataset_size'] // experiment_info['script_parameters']['batchsize']
    total_number_of_updates = updates_in_epoch * experiment_info['script_parameters']['epochs']

    # 3. extract patterns and store in .pt files from that amount of .pt snapshots
    storage_window_filenames = get_filenames_to_process(
        args.samplerate, 
        comparisons, 
        batch_size, 
        batch_idx, 
        updates_in_epoch, 
        total_number_of_updates
    ) 

    processing_window_filenames = storage_window_filenames

    while storage_window_filenames != None:
        func = lambda filename: cache_pattern_matrix(
            filename,
            args.input,
            pattern_data,
            model_cls)
        
        # map(func, storage_window_filenames)
        for x in storage_window_filenames: func(x)

        func = lambda filename: process_statistics(
            filename,
            output,
            test_loader,
            args.input,
            model_cls,
            get_prev(filename, comparisons[0], args.samplerate, updates_in_epoch, batch_size),
            get_prev(filename, comparisons[1], args.samplerate, updates_in_epoch, batch_size)
        )
        
        [func(x) for x in processing_window_filenames]

        batch_idx += 1

        # 5. delete a windows worth of .bin files from the start that aren't used anymore and process a windows worth more updates
        storage_window_filenames_old = storage_window_filenames
        storage_window_filenames = get_filenames_to_process(
            args.samplerate, 
            comparisons, 
            batch_size, 
            batch_idx, 
            updates_in_epoch, 
            total_number_of_updates
        )  

        if (storage_window_filenames != None):
            delete_unused(storage_window_filenames_old, storage_window_filenames, filename)  # delete files that exist in old that aren't in new
            processing_window_filenames = storage_window_filenames[batch_size:]

