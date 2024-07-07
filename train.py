import yaml

import numpy as np

from tqdm import trange

import ANN

config = yaml.safe_load(open('config.yaml'))
model = ANN.ANN([])
model.load(config['model'])

X_train = np.load('MNIST/train_images.npy').reshape(60000, 784)
Y_train = np.load('MNIST/train_labels.npy').reshape(60000, 1)
Y_train = np.apply_along_axis(lambda x: [1 if i == x else 0 for i in range(10)], 1, Y_train)

X_test = np.load('MNIST/test_images.npy').reshape(10000, 784)
Y_test = np.load('MNIST/test_labels.npy').reshape(10000, 1)


train_acc = np.sum(np.argmax(model(X_train), 1) == np.argmax(Y_train, 1)) / 600
test_acc = np.sum(np.argmax(model(X_test), 1).reshape(10000, 1) == Y_test) / 100

if config['batch_size'] == -1:
    for _ in trange(config['epochs']):
        model.train(X_train, Y_train, lr=1e-4)
    
elif config['batch_size'] == 1:
    for i in trange(config['epochs'] * 60000):
        model.train(X_train[i].reshape(1, 784), Y_train[i].reshape(1, 10), lr=1e-2)

train_acc = np.sum(np.argmax(model(X_train), 1) == np.argmax(Y_train, 1)) / 600
test_acc = np.sum(np.argmax(model(X_test), 1).reshape(10000, 1) == Y_test) / 100
print(train_acc, test_acc)

model.save(config['model'])