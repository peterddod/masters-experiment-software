import torch
import random
import numpy as np  
from torch import nn


class FileWriter():
    """
    Create a new file and append strings line-by-line to it with object calls.
    Initiated with a header string, which is the first string added to the file.
    Overwrite's files with the same name.
    Files not given any extension.
    """
    def __init__(self, filename, header):
        self.filename = filename
        self.header = header

        with open(f'{self.filename}', 'w') as f:
            f.write(header + '\n')
            f.close()

    def __call__(self, info):
        with open(f'{self.filename}', 'a') as f:
            f.write(info + '\n')
            f.close()


class ByteWriter():
    """
    Create a new file and append bytes with object calls.
    Overwrite's files with the same name.
    Files not given any extension.
    """
    def __init__(self, filename):
        self.filename = filename

        with open(f'{self.filename}', 'wb') as f:
            f.close()

    def __call__(self, bytes):
        with open(f'{self.filename}', 'ab') as f:
            f.write(bytes)
            f.close()


class ByteReader():
    """
    Reads bytes from a given path.
    """
    def __init__(self, filename):
        self.filename = filename

        with open(f'{self.filename}', 'wb') as f:
            f.close()

    def __call__(self, bytes):
        with open(f'{self.filename}', 'ab') as f:
            f.write(bytes)
            f.close()
            

def resetseed(seed=0):
    """
    Resets random seed for torch, random, and numpy random.
    If no seed given, then a seed of 0 is used.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def ceil_to_factor(value, factor):
    """
    Round a value to the nearest factor of a given number.
    """
    return int(np.ceil(value/factor)*factor)


def init_weights(m, method, params_dict={}, module_types=[nn.Linear, nn.LazyLinear, nn.Conv2d, nn.LazyConv2d]):
    if type(m) in module_types:
        method(m.weight, **params_dict)


def he_init(m):
    init_weights(m, nn.init.kaiming_normal_, params_dict={'mode':'fan_in', 'nonlinearity':'relu'})