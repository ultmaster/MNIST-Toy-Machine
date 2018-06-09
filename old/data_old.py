import gzip

import numpy as np
import pickle
import os

import sys

from cache import cached

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


def convert_images_to_features(images):
    features = {}
    for key in range(row_num * col_num):
        features[str(key)] = images[:, key]
    return features


@cached("train-images-flatten-pickle")
def train_image_flatten():
    train_images = []
    for _ in range(train_size):
        train_images.append(next_image(f).flatten())
    return np.asarray(train_images)


@cached("train-image-feature-dict-pickle")
def train_image_feature_dict(train_images):
    return convert_images_to_features(train_images)


@cached("test-images-flatten-pickle")
def test_image_flatten():
    test_images = []
    for _ in range(test_size):
        test_images.append(next_image(f).flatten())
    return np.asarray(test_images)


@cached("test-image-feature-dict-pickle")
def test_image_feature_dict(test_images):
    return convert_images_to_features(test_images)


os.makedirs("tmp", exist_ok=True)

with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    assert next_int(f) == 2051
    train_size = next_int(f)
    row_num = next_int(f)
    col_num = next_int(f)

    train_images = train_image_flatten()
    train_images_features = train_image_feature_dict(train_images)

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
    assert row_num == next_int(f)
    assert col_num == next_int(f)

    test_images = test_image_flatten()
    test_images_features = test_image_feature_dict(test_images)

    assert len(test_images) == test_size

with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    assert next_int(f) == 2049
    assert test_size == next_int(f)
    test_labels = []
    for _ in range(test_size):
        test_labels.append(next_char(f))
    assert len(test_labels) == test_size
