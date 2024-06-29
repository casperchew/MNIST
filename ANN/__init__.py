import scipy

import numpy as np

np.random.seed(0)
np.seterr(all='raise')

def acc(model, X, y):
	return np.sum(np.argmax(model(X), axis=1) == np.argmax(y, axis=1)) / X.shape[0] * 100

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoidPrime(x):
	return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
	return scipy.special.softmax(x, axis=1)

class ANN:
	def __init__(self, i, h, o, lr=1e-3):
		self.W_ih = np.random.randn(i, h) / 784
		self.B_ih = np.linspace(-1 / 784, 1 / 784, h)
		self.W_ho = np.random.randn(h, o) / 784
		self.B_ho = np.linspace(-1 / 784, 1 / 784, o)

		self.lr = lr
	
	def __call__(self, X):
		return sigmoid(X @ self.W_ih + self.B_ih) @ self.W_ho + self.B_ho
		# return softmax(S(X @ self.W_ih + self.B_ih) @ self.W_ho + self.B_ho)
	
	def train(self, X, Y):
		H = sigmoid(X @ self.W_ih + self.B_ih)
		YHat = self(X)

		self.W_ih -= self.lr * X.T @ (-2 * (Y - YHat) * sigmoidPrime(H @ self.W_ho + self.B_ho) @ self.W_ho.T * sigmoidPrime(X @ self.W_ih + self.B_ih)) / X.shape[0]
		self.W_ho -= self.lr * H.T @ (-2 * (Y - YHat) * sigmoidPrime(H @ self.W_ho + self.B_ho)) / X.shape[0]
	
	def save(self, filename):
		np.save(f'{filename}.npy', np.array([self.W_ih, self.B_ih, self.W_ho, self.B_ho], dtype=object))
	
	def load(self, filename):
		self.W_ih, self.B_ih, self.W_ho, self.B_ho = list(np.load(f'{filename}.npy', allow_pickle=True))