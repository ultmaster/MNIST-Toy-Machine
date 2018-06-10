import time
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import mnist
from util import Timer


def process(image):
    xf = image.flatten()
    for x in np.nditer(xf, op_flags=['readwrite']):
        x[...] = 0 if x < 160 else 1
    return xf


def go(pca_enabled=False, centralize=False):
    print("PCA:", pca_enabled)
    print("Centralize:", centralize)

    train_x, train_y = mnist.train()
    test_x, test_y = mnist.test()

    if centralize:
        train_x = mnist.train_32()
        test_x = mnist.test_32()

    train_x = train_x.reshape((train_x.shape[0], -1))
    test_x = test_x.reshape((test_x.shape[0], -1))

    if pca_enabled:
        with Timer("PCA"):
            pca = PCA(n_components=50, whiten=True)
            train_x = pca.fit_transform(train_x)
            test_x = pca.transform(test_x)

    with Timer("train"):
        max_iter = 1000 if centralize or pca_enabled else 200
        clf = MLPClassifier(max_iter=max_iter, verbose=True)
        clf.fit(train_x, train_y)
        print("Accuracy:", clf.score(test_x, test_y))


for p in [False, True]:
    for c in [False, True]:
        go(p, c)
