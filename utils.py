import torch
import random
import numpy as np  
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image


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


class Cutout(object):
    """
    Implements Cutout regularization as proposed by DeVries and Taylor (2017), https://arxiv.org/pdf/1708.04552.pdf.
    """

    def __init__(self, num_cutouts, size, p=0.5):
        """
        Parameters
        ----------
        num_cutouts : int
            The number of cutouts
        size : int
            The size of the cutout
        p : float (0 <= p <= 1)
            The probability that a cutout is applied (similar to keep_prob for Dropout)
        """
        self.num_cutouts = num_cutouts
        self.size = size
        self.p = p

    def __call__(self, img):

        height, width = img.size

        cutouts = np.ones((height, width))

        if np.random.uniform() < 1 - self.p:
            return img

        for i in range(self.num_cutouts):
            y_center = np.random.randint(0, height)
            x_center = np.random.randint(0, width)

            y1 = np.clip(y_center - self.size // 2, 0, height)
            y2 = np.clip(y_center + self.size // 2, 0, height)
            x1 = np.clip(x_center - self.size // 2, 0, width)
            x2 = np.clip(x_center + self.size // 2, 0, width)

            cutouts[y1:y2, x1:x2] = 0

        cutouts = np.broadcast_to(cutouts, (3, height, width))
        cutouts = np.moveaxis(cutouts, 0, 2)
        img = np.array(img)
        img = img * cutouts
        return Image.fromarray(img.astype('uint8'), 'RGB')


def get_train_loaders(dataset, batchsize=64, seed=1, n=None):
    if dataset == 'mnist':
        torch.manual_seed(seed)
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
        random.seed(seed)
        if n!= None:
            indices = random.sample(range(0, 50000), n)
        else:
            indices = range(0, 50000)
        torch.manual_seed(seed)
        train_loader = torch.utils.data.DataLoader(
            Subset(datasets.CIFAR10('./dataset', train=True, download=True,
                transform=transforms.Compose([
                    Cutout(num_cutouts=2, size=8, p=0.8),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])), indices),
            batch_size=batchsize, 
            shuffle=True,
        )

        random.seed(seed)
        if n!= None:
            indices = random.sample(range(0, 10000), n)
        else:
            indices = range(0, 10000)

        test_loader = torch.utils.data.DataLoader(
            Subset(datasets.CIFAR10('./dataset', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])), indices),
            batch_size=250, 
            shuffle=False,
        )

        return train_loader, test_loader
    
    elif dataset == 'cifar10-big':
        size = (128,128)
        random.seed(seed)
        if n!= None:
            indices = random.sample(range(0, 50000), n)
        else:
            indices = range(0, 50000)
        torch.manual_seed(seed)
        train_loader = torch.utils.data.DataLoader(
            Subset(datasets.CIFAR10('./dataset', train=True, download=True,
                transform=transforms.Compose([
                    Cutout(num_cutouts=2, size=8, p=0.8),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])), indices),
            batch_size=batchsize, 
            shuffle=True,
        )

        random.seed(seed)
        if n!= None:
            indices = random.sample(range(0, 10000), n)
        else:
            indices = range(0, 10000)

        test_loader = torch.utils.data.DataLoader(
            Subset(datasets.CIFAR10('./dataset', train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])), indices),
            batch_size=250, 
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
        return torch.Tensor(datasets.CIFAR10('./dataset', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])).data.reshape([50000,3,32,32])).to(torch.float32)