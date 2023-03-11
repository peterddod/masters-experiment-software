import numpy as np
import torch
from torch import nn
from time import time
from torch import multiprocessing as mp
from train import evaluate_epoch
from functools import partial
from patternanalysis import BinaryPathTree
import os


def make_activation_matrix(byte_sequence, neuron_count):
    """
    Create a `torch.Tensor` matrix for the activation of each neuron in a
    neural network per dataset sample.
    """
    bytes_per_pattern = int(np.ceil(neuron_count/8))

    byte_array = [byte_sequence[i:i+bytes_per_pattern] for i in range(0,len(byte_sequence),bytes_per_pattern)]
    bit_string_array = [ "".join(list((map(lambda y: '{0:08b}'.format(y), x))))[:neuron_count] for x in byte_array ]
    int_array = [int(x) for x in list("".join(bit_string_array))]
    activation_matrix = torch.Tensor(int_array).reshape(-1,neuron_count)

    return activation_matrix


def get_filenames_to_process(samplerate, comparisons, batch_size, batch_idx, updates_in_epoch, total_number_of_updates):
    """
    Find the file names required to process a batch_idx in file_process.py.
    Returns None item in list if batch exceeds updates_in_epochs, 
    and returns None once processing window passess all updates.
    """
    buffer = max(comparisons) + batch_size

    storage_start_point = batch_idx * batch_size
    processing_start_point = storage_start_point + max(comparisons)
    processing_end_point = storage_start_point + buffer

    if processing_start_point >= total_number_of_updates//samplerate:
        return None

    output_list = []

    while storage_start_point < processing_end_point:
        if storage_start_point >= total_number_of_updates//samplerate:
            output_list.append(None)
        else:
            epoch = (storage_start_point*samplerate) // updates_in_epoch + 1
            update = (storage_start_point*samplerate) % updates_in_epoch

            output_list.append(f'{epoch}_{update}')

        storage_start_point += 1

    return output_list


def extract(model, data):
    """
    Extract activation patterns from model using data and return as
    torch tensor.
    """
    model.eval()

    _output = []

    def hook(model, input, output):
        _output.append(output)
        return

    model.apply_activation_hook(hook)

    model(data)

    _output = torch.hstack(_output).detach()

    _output[_output!=0] = 1

    return _output


def cache_pattern_matrix(filename, input_name, data, model_cls):
    """
    Produces activation patterns matrix from a saved model when using a dataset and saves
    a .pt to the specifed path.
    """
    if filename==None:  return

    model_path = f'./results/{input_name}/snapshots/{filename}.pt'
    cache_path = f'./processed/{input_name}/.cache/{filename}.pt'

    model = model_cls()
    model.load_state_dict(torch.load(model_path))

    # find activation pattern matrix
    activation_matrix = extract(model, data)

    # cache activation pattern
    torch.save(activation_matrix, cache_path)


def get_number_of_unique_patterns(activation_matrix):
    tree = BinaryPathTree()

    [tree.add(list(x)) for x in activation_matrix]

    return len(tree)  # returns number of leaf nodes in binary tree of patterns, i.e. number of unique patterns


def cos_sim(a, b):
    return torch.dot(a, b)/(torch.linalg.norm(a)*torch.linalg.norm(b))


def get_similarities(m1, m2, compareFunction=cos_sim):
    if m1==None or m2==None: return [0]
    if m1.shape != m2.shape: return [0]

    output = []

    for i in range(m1.shape[0]):
        output.append(compareFunction(m1[i], m2[i]))

    # def func(x): output.append(compareFunction(m1[x], m2[x]))
    # map(func, range((m1.shape[0])))

    return output


def load(inputname, filename):
    output = None
    try:
        output = torch.load(f'./processed/{inputname}/.cache/{filename}.pt')
    except Exception as e:
        print(e)
        pass

    return output


def get_prev(current_filename, n, samplerate, updates_in_epoch, batch_size):
    """
    Find the previous n filename if exists, otherwise return None.
    """
    split = current_filename.split('_')
    epoch = int(split[0])
    batch_idx = int(split[1])

    current_update = (((epoch-1) * updates_in_epoch) + batch_idx)
    next_update = current_update - n*samplerate
    if next_update < 0:
        return None


    epoch = (next_update) // updates_in_epoch + 1
    update = (next_update) % updates_in_epoch

    return f'{epoch}_{update}'


def process_statistics(filename, output, test_dataloader, input_name, model_cls, at_10_filename, at_100_filename):
    """
    Processes the statistics for a particular update.

    Inputs a queue and adds a .csv string in the following order:
    - epoch
    - batch_idx
    - test_acc
    - unique_ap
    - mean_ap_sim@10
    - std_ap_sim@10
    - mean_ap_sim@100
    - std_ap_sim@100
    """
    split = filename.split('_')
    epoch = int(split[0])
    batch_idx = int(split[1])

    model = model_cls()
    model.load_state_dict(torch.load(f'./results/{input_name}/snapshots/{filename}.pt'))

    test_acc = evaluate_epoch(model, test_dataloader)
    activation_matrix = torch.load(f'./processed/{input_name}/.cache/{filename}.pt').detach()
    unique_ap = get_number_of_unique_patterns(activation_matrix)

    at_10 = load(input_name, at_10_filename)
    at_100 = load(input_name, at_100_filename)

    sim_at_10 = get_similarities(at_10, activation_matrix)
    sim_at_100 = get_similarities(at_100, activation_matrix)

    output(",".join([
        str(epoch), 
        str(batch_idx),
        str(test_acc),
        str(unique_ap),
        str(np.mean(sim_at_10)),
        str(np.std(sim_at_10)),
        str(np.mean(sim_at_100)),
        str(np.std(sim_at_100)),
        ]))


def delete_unused(old, new, input_name):
    [os.remove(f'./processed/{input_name}/.cache/{filename}.pt') for filename in set(old) - set(new) if filename != None]