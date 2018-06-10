import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import mnist
from util import Timer


def process(mat):
    return np.rint(mat / 256).astype(np.uint8)


def go(proc, central):
    print("kNN")
    print("0-1", proc)
    print("central", central)

    train_x, train_y = mnist.train()
    test_x, test_y = mnist.test()

    if central:
        train_x = mnist.train_32()
        test_x = mnist.test_32()

    if proc:
        with Timer("process"):
            train_x = process(train_x)
            test_x = process(test_x)

    train_x = train_x.reshape((train_x.shape[0], -1))
    test_x = test_x.reshape((test_x.shape[0], -1))

    with Timer("kNN fit"):
        neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        neigh.fit(train_x, train_y)

    with Timer("kNN test"):
        print("Accuracy:", neigh.score(test_x, test_y))



def go_svm(proc, pca_enabled, central):
    print("SVM")
    print("0-1", proc) # only 0 and 1
    print("Central", central) # 32x32
    print("PCA", pca_enabled) # PCA to 50 dims

    train_x, train_y = mnist.train()
    test_x, test_y = mnist.test()

    if central:
        train_x = mnist.train_32()
        test_x = mnist.test_32()

    if proc:
        with Timer("process"):
            train_x = process(train_x)
            test_x = process(test_x)

    train_x = train_x.reshape((train_x.shape[0], -1))
    test_x = test_x.reshape((test_x.shape[0], -1))

    if pca_enabled:
        with Timer("PCA"):
            pca = PCA(n_components=50, whiten=True)
            train_x = pca.fit_transform(train_x)
            test_x = pca.transform(test_x)

    with Timer("train"):
        clf = svm.SVC(cache_size=7000)
        clf.fit(train_x, train_y)
        print("Accuracy:", clf.score(test_x, test_y))



# go(True, False)
# go(False, True)
# go(True, True)
#
# go_svm(True, True, False)
# go_svm(False, True, True)
# go_svm(True, False, True)
# go_svm(False, False, True)
go_svm(True, True, True)
