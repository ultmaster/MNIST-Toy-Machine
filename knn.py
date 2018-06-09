from sklearn.neighbors import KNeighborsClassifier
import mnist


neigh = KNeighborsClassifier(n_neighbors=3)
train_x, train_y = mnist.train()
test_x, test_y = mnist.test()

train_x = train_x.reshape((train_x.shape[0], -1))
test_x = test_x.reshape((test_x.shape[0], -1))

neigh.fit(train_x, train_y)
print(neigh.score(test_x, test_y))