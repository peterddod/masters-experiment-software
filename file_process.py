"""
Script to process raw byte data produced by main.py.

Part of MSci Project for Peter Dodd @ University of Glasgow.
"""
import argparse
from datetime import datetime
import json
import numpy as np
from activation import make_activation_matrix
from utils import FileWriter, ceil_to_factor
import torch.multiprocessing as mp
from config import *


parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", default='test', help="Name of input folder from /results")
parser.add_argument("-f", "--filename", default=None, help="Name of output file")  # TODO: get default value from experiment.info

parser.add_argument("-s", "--samplerate", default=10, help="Gap between gradient updates for comparisons", type=int)


if __name__ == '__main__':
    args = parser.parse_args()

    # read experiment.info file to get activation layer sizes
    with open(f'./results/{args.input}/experiment.info', 'r') as f:
        experiment_info = json.load(f)
        f.close()

    filename = args.filename  # TODO: add folder creation for if /processed doesn't exist

    # create output file
    output = FileWriter(f'./processed/{filename}.csv', ",".join(['epoch','batch_idx','test_acc','unique_ap','mean_ap_sim@10','std_ap_sim@10','mean_ap_sim@100','std_ap_sim@100']))

    # TODO: if there is none, create .cache folder in ./processed to store .bin files of patterns
    
    # 1. find range of updates required to process one set of stats  NOTE: not all updates have all stats due to each update potentially requiring earlier and later updates to attain values
    model_cls = model_map[experiment_info['script_parameters']['model']]

    comparisons = [1,10]  # how many update steps between comparisons
    stop_file = None  # TODO: filename of when to stop processing files
    batch_size = 10
    batch_idx = 0

    write_to_csv_queue = mp.Queue()

    # 3. extract patterns and store in .bin files from that amount of .pt snapshots
    storage_window_filenames = get_filenames_to_process(args.samplerate, comparisons, batch_size, batch_idx)  # TODO: create list of filenames for currently stored updates
    processing_window_filenames = storage_window_filenames

    while storage_window_filenames != None:
        pool = mp.Pool(4)
        r = pool.map_async(lambda x: cache_pattern_matrix(x, model_cls), storage_window_filenames)  # TODO: create cache_pattern_matrix function, if file exists or parameter=None then skip
        r.wait()

        # 4. process stastics for window and append to file
        r = pool.map_async(lambda x: process_stastics(x, comparisons, args.samplerate, write_to_csv_queue), processing_window_filenames)
        r.wait()

        # write results to output file
        while write_to_csv_queue.qsize():
            output(write_to_csv_queue.get())

        batch_idx += 1

        # 5. delete a windows worth of .bin files from the start that aren't used anymore and process a windows worth more updates
        storage_window_filenames_old = storage_window_filenames
        storage_window_filenames = get_filenames_to_process(args.samplerate, comparisons, batch_size, batch_idx)  # TODO: create list of filenames for currently stored updates, returns null's if updates don't fit batch size

        delete_unused(storage_window_filenames_old, storage_window_filenames) # delete files that exist in old that aren't in new

        if (storage_window_filenames != None):
            processing_window_filenames = storage_window_filenames[:batch_size]