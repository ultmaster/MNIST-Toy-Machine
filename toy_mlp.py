from sklearn import svm
from sklearn.neural_network import MLPClassifier

from read_data import *

def score(a, b):
    assert len(a) == len(b)
    return sum(map(lambda x: int(a[x] == b[x]), range(len(a)))) / len(a)


def process(image):
    xf = image.flatten()
    for x in np.nditer(xf, op_flags=['readwrite']):
        x[...] = 0 if x < 160 else 1
    return xf


X = []
for idx, x in enumerate(train_images[:10000]):
    X.append(process(x))
    if idx % 1000 == 0:
        print(idx)
# print('ok')
clf = MLPClassifier()
clf.fit(X, train_labels[:10000])

X = []
for x in test_images:
    X.append(process(x))
print(score(clf.predict(X), test_labels))
