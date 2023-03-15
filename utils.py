import torch
import random
import numpy as np  
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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


def get_train_loaders(dataset, batchsize=64):
    if dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./dataset', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=batchsize, 
            shuffle=True,
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./dataset', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=256, 
            shuffle=False,
        )

        return train_loader, test_loader
    
    elif dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./dataset', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=batchsize, 
            shuffle=True,
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./dataset', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=256, 
            shuffle=False,
        )

        return train_loader, test_loader
    

def get_pattern_data(dataset):
    if dataset == 'mnist':
        return datasets.MNIST('./dataset', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])).data.reshape([60000,1,28,28]).to(torch.float32)
    
    elif dataset == 'cifar10':
        return datasets.CIFAR10('./dataset', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])).data.reshape([50000,3,28,28]).to(torch.float32)