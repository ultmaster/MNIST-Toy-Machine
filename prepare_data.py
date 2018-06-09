import os
import shutil
from zipfile import ZipFile

import numpy as np
from PIL import Image

os.makedirs("tmp", exist_ok=True)
os.makedirs("tmp/images", exist_ok=True)

with ZipFile('data/mnist.zip') as myzip:
    myzip.extractall("tmp")

for t in ["test", "train"]:
    for f in ["data", "label"]:
        shutil.move("tmp/mnist/mnist_{t}/mnist_{t}_{f}".format(t=t, f=f), "tmp/mnist_%s_%s" % (t, f))


def resize_features(feature_name):
    features = np.fromfile("tmp/%s" % feature_name, dtype=np.uint8).reshape((-1, 45, 45))
    N = features.shape[0]
    WIDTH = features.shape[1]
    target_width = 32
    features_x = []
    for i, row in enumerate(features):
        med = np.median(np.asarray([[j, k] for j in range(WIDTH) for k in range(WIDTH) if row[j][k] > 160]), axis=0)
        nx = int(min(max(med[0] - target_width / 2, 0), WIDTH - target_width))
        ny = int(min(max(med[1] - target_width / 2, 0), WIDTH - target_width))
        features_x.append(row[nx:nx + target_width, ny:ny + target_width])
        if i < 100:
            im = Image.fromarray(features_x[-1])
            im.save("tmp/images/example%d.png" % i)
        if i % 1000 == 0:
            print("Step: %d / %d" % (i, N))
    np.save("tmp/%s_32" % feature_name, np.asarray(features_x))


resize_features("mnist_test_data")
resize_features("mnist_train_data")
