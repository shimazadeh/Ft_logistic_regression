import numpy as np
import math
import pandas as pd

class MyLogisticRegression():
	def __init__(self, theta=None , alpha=0.001, max_iter=1000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = theta
		
	@staticmethod
	def add_intercept(x):
		if x is None:
			return 0
		df = pd.DataFrame(x)
		num_columns = len(df.axes[1])
		first_column = df.pop(0)
		df.insert(0, 0, 1)

		i = 1
		while(i < num_columns + 1):
			if (i < num_columns):
				next_column = df.pop(i)
			else:
				next_column = 0
			df.insert(i, i, first_column)
			first_column = next_column
			i = i + 1
		x = df.to_numpy()
		return (x)

	def sigmoid_(self, x):#for two features only
		return (1 / (1 + np.exp(-x)))

	def softmax(self, x):#for more than two features
		exp_x = np.exp(x)
		return exp_x / np.sum(exp_x, axis=1, keepdims=True)

	def logistic_predict_(self, x):
		x_ = self.add_intercept(x)
		res = np.dot(x_, self.theta)
		# sig_ = sigmoid_(res)
		sig_ = softmax(res)
		return (sig_)

	def log_gradient(self, x, y):
		x_ = self.add_intercept(x)
		h_ = self.logistic_predict_(x)
		m = len(x)
		grad= np.dot(x_.T, (h_ - y))/m
		return (grad)

	def cost(self, x, y):
		x_ = self.add_intercept(x)
		y_ = self.logistic_predict_(x)
		m = len(x)
		cost = (-1 / m) * (np.sum(y * np.log(y_)))
		return (cost)
	
	def fit(self, x, y):
		x_ = self.add_intercept(x)
		self.theta = np.zeros((x_.shape[1], y.shape[1]))

		for _ in range(self.max_iter):
			grad = self.log_gradient(x, y)
			self.theta -= self.alpha * grad

	def predict(self, x):
		x_ = self.add_intercept(x)
		probabilities = self.logistic_predict_(x)
		return np.argmax(probabilities, axis=1)