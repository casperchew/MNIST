# MNIST

## Setup

### Installing requirements
```
\MNIST>pip install -r requirements.txt
```

### MNIST dataset
- Create a folder `MNIST`
- Move 4 dataset files into `\MNIST`:
	- `train-images-idx3-ubyte.gz`
	- `train-labels-idx1-ubyte.gz`
	- `t10k-images-idx3-ubyte.gz`
	- `t10k-labels-idx1-ubyte.gz`
- Run `idx_to_npy.py` to convert dataset into numpy arrays

## Usage

1) Create `models` directory
```
\>mkdir models
```
2) Create a neural network model using `model.py`
```
\>py model.py
```
3) Set config in `config.yaml`
```
model: {string}
epochs: {int}
batch_size: {int}
```
4) Test using `test.py` or train using `train.py`
```
\>py test.py
\>py train.py
```

## Models

### K-NN
`k=1`:
- Error rate: 3.1%
- Recognition time on test dataset: 1h 1min 9.5s

### Linear
- Error rate: 24.31%
- Training config: 30 epochs, batch_size=1, lr=1e-3
- Training time: 2min 51s
