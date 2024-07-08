import numpy as np

import kNN

X_train = np.load('MNIST/train_images.npy').reshape(60000, 784)
Y_train = np.load('MNIST/train_labels.npy').reshape(60000, 1)
X_test = np.load('MNIST/test_images.npy').reshape(10000, 784)
Y_test = np.load('MNIST/test_labels.npy').reshape(10000, 1)

model = kNN.kNN(X_train, Y_train)

print(sum(model(X_test) == Y_test))