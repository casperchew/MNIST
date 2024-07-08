import numpy as np

class kNN:
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y
	
	def __call__(self, X):
		return np.apply_along_axis(lambda row : self.Y[np.argmin(np.apply_along_axis(lambda x : np.linalg.norm(x - row), 1, self.X))], 1, X)