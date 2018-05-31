import gzip

import numpy as np
import pickle
import os

import sys


"""
Use
    from read_data import *
to get all the data
"""


def next_int(fin):
    return int.from_bytes(fin.read(4), byteorder='big')


def next_char(fin):
    return int.from_bytes(fin.read(1), byteorder='big')


def next_image(fin):
    s = np.zeros([row_num, col_num], dtype=np.uint8)
    for i in range(row_num):
        for j in range(col_num):
            s[i][j] = next_char(fin)
    return s


os.makedirs("tmp", exist_ok=True)

with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    assert next_int(f) == 2051
    train_size = next_int(f)
    train_images = []
    row_num = next_int(f)
    col_num = next_int(f)
    pickle_path = "tmp/train-images-pickle"
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            train_images = pickle.load(f)
    else:
        print("Train image cache not found. Rebuilding...", file=sys.stderr)
        for _ in range(train_size):
            train_images.append(next_image(f))
        with open(pickle_path, "wb") as f:
            pickle.dump(train_images, f)
    assert len(train_images) == train_size

with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    assert next_int(f) == 2049
    assert train_size == next_int(f)
    train_labels = []
    for _ in range(train_size):
        train_labels.append(next_char(f))
    assert len(train_labels) == train_size

with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    assert next_int(f) == 2051
    test_size = next_int(f)
    test_images = []
    assert row_num == next_int(f)
    assert col_num == next_int(f)
    pickle_path = "tmp/test-images-pickle"
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            test_images = pickle.load(f)
    else:
        print("Test image cache not found. Rebuilding...", file=sys.stderr)
        for _ in range(test_size):
            test_images.append(next_image(f))
        with open(pickle_path, "wb") as f:
            pickle.dump(test_images, f)
    assert len(test_images) == test_size

with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    assert next_int(f) == 2049
    assert test_size == next_int(f)
    test_labels = []
    for _ in range(test_size):
        test_labels.append(next_char(f))
    assert len(test_labels) == test_size
