import gzip
import numpy as np

from tqdm import tqdm

train_images = gzip.open('MNIST/train-images-idx3-ubyte.gz')
train_labels = gzip.open('MNIST/train-labels-idx1-ubyte.gz')
test_images = gzip.open('MNIST/t10k-images-idx3-ubyte.gz')
test_labels = gzip.open('MNIST/t10k-labels-idx1-ubyte.gz')

magic_number = train_images.read(4)
n_dims = int(magic_number.hex()[6:8])
dims = [int(train_images.read(4).hex(), 16) for _ in range(n_dims)]
data = np.array([int(train_images.read(1).hex(), 16) for _ in tqdm(range(np.prod(dims)))]).reshape(dims)
np.save('MNIST/train_images', data)

magic_number = train_labels.read(4)
n_dims = int(magic_number.hex()[6:8])
dims = [int(train_labels.read(4).hex(), 16) for _ in range(n_dims)]
data = np.array([int(train_labels.read(1).hex(), 16) for _ in tqdm(range(np.prod(dims)))]).reshape(dims)
np.save('MNIST/train_labels', data)

magic_number = test_images.read(4)
n_dims = int(magic_number.hex()[6:8])
dims = [int(test_images.read(4).hex(), 16) for _ in range(n_dims)]
data = np.array([int(test_images.read(1).hex(), 16) for _ in tqdm(range(np.prod(dims)))]).reshape(dims)
np.save('MNIST/test_images', data)

magic_number = test_labels.read(4)
n_dims = int(magic_number.hex()[6:8])
dims = [int(test_labels.read(4).hex(), 16) for _ in range(n_dims)]
data = np.array([int(test_labels.read(1).hex(), 16) for _ in tqdm(range(np.prod(dims)))]).reshape(dims)
np.save('MNIST/test_labels', data)