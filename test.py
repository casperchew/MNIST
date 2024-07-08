import yaml

import numpy as np

import NN

config = yaml.safe_load(open('config.yaml'))

model = NN.NN([])
model.load(config['model'])

X_train = np.load('MNIST/train_images.npy').reshape(60000, 784)
Y_train = np.load('MNIST/train_labels.npy').reshape(60000, 1)
X_test = np.load('MNIST/test_images.npy').reshape(10000, 784)
Y_test = np.load('MNIST/test_labels.npy').reshape(10000, 1)

train_acc = np.sum(np.argmax(model(X_train), 1).reshape(60000, 1) == Y_train) / 600
test_acc = np.sum(np.argmax(model(X_test), 1).reshape(10000, 1) == Y_test) / 100
print(train_acc, test_acc)