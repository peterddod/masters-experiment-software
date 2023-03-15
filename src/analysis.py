import functools
import torch
from utils import *
from torch import nn


def analyse_model(model, data, file):
    """
    Analyse a model to find the indexes in model.modules() that output activated tensors, 
    i.e. tensors in a neural network that have passed through an activation function. 
    
    Also finds indexes of model.modules() that can be skipped when iterating through and forward
    passing. Currently only recognises nn.ReLU.

    Contains an input for a ByteWriter to append the found patterns to a file.

    Returns a tuple containing (activation_idxs, skip_idxs, layer_output_sizes).
    """
    model.eval()

    # find where to get APs from
    activation_idxs = []
    skip_idxs = []
    layer_output_sizes = []

    data_test = data[0].reshape([1,*data.shape[1:]])

    for i, module in enumerate(model.modules()):
        if isinstance(module, (nn.Flatten, nn.LogSoftmax, nn.Linear, nn.LazyLinear)):
            data_test = module(data_test)
            continue
        elif isinstance(module, (nn.ReLU)):
            data_test = module(data_test)
            activation_idxs.append(i)
            layer_output_sizes.append(list(data_test.shape[1:]))
        else:
            skip_idxs.append(i)

    patterns = get_patterns_from_dataset(model, skip_idxs, activation_idxs, data)

    for pattern in patterns:
        file(pattern)

    return activation_idxs, skip_idxs, layer_output_sizes


def extract(state_dict, model_cls, skip_idxs, activation_idxs, data, file):
    """
    Extract the activation patterns from a model over a dataset and append to a file.

    Inputs a state_dict and model_cls to create a copy model to allow for concurrency with
    training.
    """
    model = model_cls()
    model.load_state_dict(state_dict)

    model.eval()

    patterns = get_patterns_from_dataset(model, skip_idxs, activation_idxs, data)

    for pattern in patterns:
        file(pattern)


def get_patterns_from_dataset(model, skip_idxs, activation_idxs, data):
    """
    Get every activation pattern produced by a model from a data tensor.

    Returns the patterns as a byte sequence.
    """
    pattern_list = map(functools.partial(
            get_activation_pattern_and_reshape, 
            model, 
            skip_idxs,
            activation_idxs,
            ), data)

    return pattern_list


def get_activation_pattern_and_reshape(model, skip_idx, activation_idxs, data):
    """
    Reshapes datapoint to appropriate size and extracts activation patterns.
    """
    data = data.reshape([1,*data.shape])

    return get_activation_pattern(model, skip_idx, activation_idxs, data)


def get_activation_pattern(model, skip_idx, activation_idxs, data):
    """
    Get the activation patterns from a model as bytes. Each character represents a neurons 
    activation in the model given the provided data point.
    
    The function uses model.modules() to get activation patterns, so a list of skip_idxs and
    weight_idxs are required.

    Returns bytes of pattern.
    """
    outputs = []

    for i, module in enumerate(model.modules()):
        if i in skip_idx:
            continue

        data = module(data)

        if i in activation_idxs:
            outputs.append(torch.flatten(data,1))

    path = torch.hstack(outputs).detach()
    
    path[path!=0] = 1

    path_encoding = encode(path.flatten())

    return path_encoding


def encode(tensor):
    """
    Encodes a tensor of 1 or 0 numbers as bytes.
    If the length of the array is not a multiple of 8, then the output
    will have trailing unused 0 bits.
    """
    count = 0
    current_value = 0
    output_list = []

    for element in tensor:
        element = int(element.item())
        current_value += element * (count ** 2)
        count += 1

        if count == 8:
            output_list.append(current_value)
            count = 0
            current_value = 0

    if count != 0:
        output_list.append(current_value)

    return bytes(output_list)