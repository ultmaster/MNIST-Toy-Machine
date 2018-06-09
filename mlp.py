import time
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import mnist


def score(a, b):
    assert len(a) == len(b)
    return sum(map(lambda x: int(a[x] == b[x]), range(len(a)))) / len(a)


def process(image):
    xf = image.flatten()
    for x in np.nditer(xf, op_flags=['readwrite']):
        x[...] = 0 if x < 160 else 1
    return xf


_, train_y = mnist.train()
_, test_y = mnist.test()
train_x = mnist.train_32()
test_x = mnist.test_32()

train_x = train_x.reshape((train_x.shape[0], -1))
test_x = test_x.reshape((test_x.shape[0], -1))


print('PCA fit start...')
t0 = time.time()
pca = PCA(n_components=50, whiten=True)
train_x_dim = pca.fit_transform(train_x)
print('Finished in %.3fs' % (time.time() - t0))

t0 = time.time()
print('Start training...')
clf = MLPClassifier()
clf.fit(train_x_dim, train_y)
print('Finished in %.3fs' % (time.time() - t0))

print('Accuracy:', score(clf.predict(pca.transform(test_x)), test_y))
