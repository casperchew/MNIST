import os
import numpy as np

from scipy.special import softmax

np.random.seed(0)
np.seterr(all='ignore')

# layers
class LinearLayer:
	def __init__(self, input_size, output_size):
		self.weights = np.random.rand(input_size, output_size) - 0.5
		self.biases = np.random.rand(output_size) - 0.5
	
	def __call__(self, X):
		self.X = X
		return X @ self.weights + self.biases
	
	def train(self, error, *args):
		self.weights -= self.X.T @ error
		self.biases -= np.average(error, 0)
		return error @ self.weights.T

# class Convolution2DLayer:
# 	# TODO
# 	def __init__(self, input_resolution, kernals):
# 		self.input_resolution = input_resolution
# 		self.kernals = kernals
# 		return
	
# 	def __call__(self, X):
# 		def convolve(x):
# 			x = x.reshape(self.input_resolution)
# 			y = []
# 			for kernal in self.kernals:
# 				output = np.empty([
# 					self.input_resolution[0] - kernal.shape[0] + 1,
# 					self.input_resolution[1] - kernal.shape[1] + 1
# 				])
# 				for i in range(output.shape[0]):
# 					for j in range(output.shape[1]):
# 						input_matrix = x[i: i + kernal.shape[0], j: j + kernal.shape[1]]
# 						output[i][j] = np.sum(input_matrix * kernal)
				
# 				y += list(output.reshape(-1))
			
# 			return y
		
# 		return np.apply_along_axis(convolve, 1, X)
	
# 	def train(self, error, *args):
# 		return error

# activation functions
class SigmoidLayer:
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

	def sigmoid_prime(x):
		return SigmoidLayer.sigmoid(x) * (1 - SigmoidLayer.sigmoid(x))

	def __init__(self):
		return
	
	def __call__(self, X):
		self.X = X
		return SigmoidLayer.sigmoid(X)
	
	def train(self, error, *args):
		return error * SigmoidLayer.sigmoid_prime(self.X)

class ReLULayer:
	ReLU = np.vectorize(lambda x: x if x > 0 else 0)
	ReLU_prime = np.vectorize(lambda x: 1 if x >= 0 else 0)

	def __init__(self):
		return
	
	def __call__(self, X):
		self.X = X
		return ReLULayer.ReLU(X)
	
	def train(self, error, *args):
		return error * ReLULayer.ReLU_prime(self.X)

class LeakyReLULayer:
	def __init__(self, leak=1e-2):
		self.LeakyReLU = np.vectorize(lambda x: x if x > 0 else leak * x)
		self.LeakyReLU_prime = np.vectorize(lambda x: 1 if x >= 0 else leak)
	
	def __call__(self, X):
		self.X = X
		return self.LeakyReLU(X)
	
	def train(self, error, *args):
		return error * self.LeakyReLU_prime(self.X)

# other layers
class SoftmaxLayer:
	def __init__(self):
		return
	
	def __call__(self, X):
		return softmax(X, 1)
	
	def train(self, error, *args):
		return error

class NN:
	def __init__(self, layers):
		self.layers = layers
		self.previous_error = 0
	
	def __call__(self, X):
		for layer in self.layers:
			X = layer(X)

		return X
	
	def train(self, X, Y, batch_size=-1, lr=1e-4, momentum_constant=0.1):
		# TODO: batch_size

		YHat = self(X)
		momentum_lr = (1 - momentum_constant) * lr
		error = momentum_constant * self.previous_error + momentum_lr * (YHat - Y)
		self.previous_error = error

		for layer in self.layers[::-1]:
			error = layer.train(error, 1)
	
	def save(self, model_name):
		try:
			for dirpath, dirnames, filenames in os.walk(f'models/{model_name}'):
				for filename in filenames:
					os.remove(f'models/{model_name}/{filename}')
			os.rmdir(f'models/{model_name}')
		except Exception as e:
			pass

		os.mkdir(f'models/{model_name}')
		np.save(f'models/{model_name}/layers', [layer.__class__.__name__ for layer in self.layers])
		for i, layer in enumerate(self.layers):
			try:
				np.save(f'models/{model_name}/{i}_weights', layer.weights)
				np.save(f'models/{model_name}/{i}_biases', layer.biases)
			except Exception as e:
				pass
	
	def load(self, model_name):
		for i, layer in enumerate(np.load(f'models/{model_name}/layers.npy')):
			try:
				self.layers.append(globals()[layer]())
			except Exception as e:
				layer = globals()[layer](*np.load(f'models/{model_name}/{i}_weights.npy').shape)
				layer.weights = np.load(f'models/{model_name}/{i}_weights.npy')
				layer.biases = np.load(f'models/{model_name}/{i}_biases.npy')
				self.layers.append(layer)