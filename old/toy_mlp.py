import time
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

t0 = time.time()
print('Importing data...')
from data import *
print('Finished in %.3fs' % (time.time() - t0))


def score(a, b):
    assert len(a) == len(b)
    return sum(map(lambda x: int(a[x] == b[x]), range(len(a)))) / len(a)


def process(image):
    xf = image.flatten()
    for x in np.nditer(xf, op_flags=['readwrite']):
        x[...] = 0 if x < 160 else 1
    return xf

print('PCA fit start...')
t0 = time.time()
pca = PCA(n_components=3, whiten=True)
train_images_dim = pca.fit_transform(train_images)
print('Finished in %.3fs' % (time.time() - t0))

t0 = time.time()
print('Start training...')
clf = MLPClassifier()
clf.fit(train_images_dim, train_labels)
print('Finished in %.3fs' % (time.time() - t0))

print('Accuracy:', score(clf.predict(pca.transform(test_images)), test_labels))
