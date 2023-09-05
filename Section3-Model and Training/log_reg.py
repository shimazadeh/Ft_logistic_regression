import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class MyLogisticRegression():
	def __init__(self, theta=None , alpha=0.001, max_iter=1000, label_encoder=None):
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = theta
		self.label_encoder = LabelEncoder()

	def add_intercept(self, x):
		if x is None:
			return 0
		df = pd.DataFrame(x)
		df.insert(0, 'intercept', 1)
		new_x = df.to_numpy()
		return new_x

	def lab_encoder(self, y):
		y_encoded = self.label_encoder.fit_transform(y)
		onehot_encoder = OneHotEncoder(sparse_output=False)
		y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))

		return (y_onehot)

	def label_decoder(self, y_encoded):
		predicted_labels = self.label_encoder.inverse_transform(y_encoded)
		return (predicted_labels)

	def sigmoid_(self, x):#for two features only
		return (1 / (1 + np.exp(-x)))

	def cost(self, x, y):
		y_ = self.logistic_predict_(x)
		m = len(x)
		cost = (-1 / m) * (np.sum(y * np.log(y_)))
		return (cost)
	
	def fit(self, x, y):
		y_ = self.lab_encoder(y.squeeze())
		m = len(x)
		self.theta = np.random.randn(x.shape[1], y_.shape[1]) * 0.01
		
		# print("theta is:", self.theta)
		for _ in range(self.max_iter):
			h_ = self.probabilities(x)
			grad= np.dot(x.T, (h_ - y_))/m
			self.theta -= self.alpha * grad
		# print("grad is:", grad)
		# print("y encoded: ", y_)

	def probabilities(self, x):
		res = np.dot(x, self.theta)
		exp_x = np.exp(res)
		sig_ = exp_x / np.sum(exp_x, axis=1, keepdims=True)#for two features: sig_ = sigmoid_(res)
		probabilities = sig_
		return probabilities

	def predict(self, x):
		probabilities = self.probabilities(x)
		y_encoded = np.argmax(probabilities, axis=1)
		y_pred = self.label_decoder(y_encoded)
		return (y_pred)

	def count(self, y, y_hat, true_val, pred_val):
		data = pd.DataFrame({'y': y, 'y_pred': y_hat})
		count = 0
		for row_name, row in data.iterrows():
			if (row[0] == true_val and row[1] == pred_val):
				count += 1
		return (count)

	def confusion_matrix(self, y, y_hat, label=None):
		unique_categories = sorted(pd.unique(y))
		df = pd.DataFrame(index=unique_categories, columns=unique_categories)
		print("total # of data: ", len(y))
		for row_name, row_data in df.iterrows():
			true_label = row_name
			for column_name, column_data in row_data.items():
				predicted_label = column_name
				df.at[true_label, predicted_label] = self.count(y, y_hat, true_label, predicted_label)

		print("the entire matrix:")
		print(df)
		if (label):
			res = df.loc[label, label]
		else:
			res = df
		print("the requested part:")
		print(res)
		return (res)
		
	def parameteres(self, y, y_hat, label=None):
		data = pd.DataFrame({'y': y, 'y_pred': y_hat})
		TP, TN, FP, FN = 0, 0, 0, 0
		for row_name, row in data.iterrows():
			if (row[0] == label and row[1] == label):
				TP += 1
			elif (row[0] == label and row[1] != label):
				FN += 1
			elif (row[0] != label and row[1] == label):
				FP += 1
			elif (row[0] != label and row[1] != label):
				TN += 1
		res = [TP, FN, FP, TN]
		return (res)

	def accuracy_score_(self, y, y_hat, label=None):
		param = self.parameteres(y, y_hat,label)
		accuracy =  (param[0] + param[3] / (param[0] + param[1] + param[2] + param[3]))
		return (accuracy)

	def precision_score_(self, y, y_hat, label=None):
		param = self.parameteres(y, y_hat,label)
		precision = param[0] / (param[0] + param[2])
		return (precision)

	def recall_score_(self, y, y_hat, label=None):
		param = self.parameteres(y, y_hat, label)
		recall = param[0] / (param[0] + param[1])
		return (recall)

	def F1_score_(self, y, y_hat, label=None):
		precision = self.precision_score_(y, y_hat, label)
		recall = self.recall_score_(y, y_hat, label)
		F1 = (2 * precision * recall) / (precision + recall)
		return (F1)