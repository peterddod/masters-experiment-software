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


parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", default='test', help="Name of input folder from /results")
parser.add_argument("-f", "--filename", default=None, help="Name of output file")  # TODO: get default value from experiment.info

parser.add_argument("-s", "--samplesize", default=1, help="Gap between gradient updates for comparisons", type=int)


if __name__ == '__main__':
    args = parser.parse_args()

    # read experiment.info file to get activation layer sizes
    with open(f'./results/{args.input}/info.json', 'r') as f:
        experiment_info = json.load(f)
        f.close()

    filename = args.filename  # TODO: add folder creation for if /processed doesn't exist

    # create output file
    output = FileWriter('./processed/{filename}.csv', ",".join(['epoch','batch_idx','test_acc','unique_ap','mean_ap_sim','std_ap_sim']))

    number_of_neurons = np.sum([np.prod(x) for x in experiment_info['layer_output_sizes']])

    # the number of bytes used to store the patterns produced by a gradient update over a dataset
    read_interval = int(np.ceil(number_of_neurons/8) * experiment_info['dataset_size'])

    with open(f'./results/{args.input}/patterns.bin', 'rb') as f:
        byte_patterns = f.read()

        for pattern_start in range(0,len(byte_patterns), read_interval):
            activation_matrix = make_activation_matrix(byte_patterns[pattern_start:pattern_start+read_interval], number_of_neurons)

            if pattern_start != 0:
                
                # compare to previous patterns for similarity

                # save similarity and number of unique patterns to file

            previous = activation_matrix

