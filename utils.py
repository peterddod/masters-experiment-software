import torch
import random
import numpy as np  


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


class ByteWriter():
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