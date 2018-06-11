import numpy as np


def train():
    features = np.fromfile("tmp/mnist_train_data", dtype=np.uint8).reshape((-1, 45, 45))
    labels = np.fromfile("tmp/mnist_train_label", dtype=np.uint8)
    return features, labels


def test():
    features = np.fromfile("tmp/mnist_test_data", dtype=np.uint8).reshape((-1, 45, 45))
    labels = np.fromfile("tmp/mnist_test_label", dtype=np.uint8)
    return features, labels


def train_32():
    return np.load("tmp/mnist_train_data_32.npy")


def test_32():
    return np.load("tmp/mnist_test_data_32.npy")


def train_32_flat_labeled(size=None):
    features = train_32().reshape((-1, 32 * 32)).astype(np.float32) / 256
    labels = np.fromfile("tmp/mnist_train_label", dtype=np.uint8)
    return features[:size], labels[:size]


def test_32_flat_labeled(size=None):
    features = test_32().reshape((-1, 32 * 32)).astype(np.float32) / 256
    labels = np.fromfile("tmp/mnist_test_label", dtype=np.uint8)
    return features[:size], labels[:size]
