import torch
import random
import numpy as np  


"""
Create a new file and append strings line-by-line to it with object calls.
Initiated with a header string, which is the first string added to the file.
Overwrite's files with the same name.
Files automatically saved as '.csv', so file type is needed.
"""
class FileWriter():
    def __init__(self, filename, header):
        self.filename = filename
        self.header = header

        with open(f'{self.filename}.csv', 'w') as f:
            f.write(header + '\n')
            f.close()

    def __call__(self, info):
        with open(f'{self.filename}.csv', 'a') as f:
            f.write(info + '\n')
            f.close()


"""
Resets random seed for torch, random, and numpy random.
If no seed given, then a seed of 0 is used.
"""
def resetseed(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)