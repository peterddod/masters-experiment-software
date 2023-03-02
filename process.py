"""
Script to process raw byte data produced by main.py.

Part of MSci Project for Peter Dodd @ University of Glasgow.
"""
import argparse
from datetime import datetime


parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", default='test', help="Name of input folder from /results")
parser.add_argument("-f", "--filename", default=f'{datetime.now().strftime("%d.%m.%Y %H.%M.%S")}', help="Name of output file")


if __name__ == '__main__':
    args = parser.parse_args()