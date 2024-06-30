import numpy as np

np.random.seed(0)

LR = 1e-3

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
	return sigmoid(x) * (1 - sigmoid(x))

# layers
class LinearLayer:
	def __init__(self, input_size, output_size):
		self.weights = np.random.rand(input_size, output_size) - 0.5
		self.biases = np.random.rand(output_size) - 0.5
	
	def __call__(self, X):
		self.X = X
		return X @ self.weights + self.biases
	
	def train(self, error):
		self.weights -= LR * (self.X.T @ error)
		return error @ self.weights.T

# activation functions
class SigmoidLayer:
	def __init__(self):
		return
	
	def __call__(self, X):
		self.X = X
		return sigmoid(X)
	
	def train(self, error):
		return error * sigmoid_prime(self.X)

# other layers
class SoftmaxLayer:
	def __init__(self):
		return
	
	def __call__(self, X):
		# TODO
		# X = np.exp(X)
		# sum_exps = np.sum(X, axis=1)
		return X
	
	def train(self, error):
		return error

class ANN:
	def __init__(self, layers, lr=1e-3):
		self.layers = layers
		self.lr = lr
	
	def __call__(self, X):
		for layer in self.layers:
			X = layer(X)

		return X
	
	def train(self, X, Y):
		YHat = self(X)
		error = YHat - Y
		for layer in self.layers[::-1]:
			error = layer.train(error)

		return error
	
	def save(self, filename):
		return
	
	def load(self, filename):
		return