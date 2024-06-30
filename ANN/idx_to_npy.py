import gzip
import numpy as np

from tqdm import tqdm

train_images = gzip.open('MNIST/train-images-idx3-ubyte.gz')

# magic number (2049/2051)
magic_number = train_images.read(4)

n_dims = int(magic_number.hex()[6:8])

dims = [int(train_images.read(4).hex(), 16) for _ in range(n_dims)]

data = np.array([int(train_images.read(1).hex(), 16) for _ in tqdm(range(np.prod(dims)))]).reshape(dims)

np.save('MNIST/train_images', data)