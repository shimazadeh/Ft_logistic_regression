import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import joblib
import json

np.random.seed(42)

class MyLogisticRegression():
	def __init__(self, theta=None , alpha=0.001, max_iter=100, batch=None, LB=LabelEncoder()):
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = theta
		self.label_encoder = LB
		self.losses = pd.DataFrame(columns=["Gryffindor", "Ravenclaw", "Slytherin", "Hufflepuff"])
		self.batch = batch

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

	def softmax_(self, x):
		z = np.dot(x, self.theta)
		exp_z = np.exp(z)
		softmax = exp_z / np.sum(exp_z, axis=1, keepdims=True)
		return (softmax)

	def loss_(self, y, y_hat):
		y_hat = y_hat  + 1e-15  # to prevent log(0) or log(1)
		loss = np.sum(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat), axis=0) / len(y)
		return (-loss)
	
	def gradient(self, x, y, y_hat):
		m = len(x)
		gradient = np.dot(x.T, (y_hat - y)) / m 
		return gradient
		
	def fit(self, x, y):
		y_ = self.lab_encoder(y.squeeze())
		self.theta = np.random.randn(x.shape[1], y_.shape[1]) * np.sqrt(2 / x.shape[1])

		if self.batch is None:
			self.batch = len(x)

		for _ in range(self.max_iter):
			indices = np.random.permutation(len(x))[:self.batch]

			x_batch = x[indices]
			y_batch = y_[indices]

			h_ = self.softmax_(x_batch)
			grad= self.gradient(x_batch, y_batch, h_)
			self.theta -= self.alpha * grad
			
			loss = self.loss_(y_batch, h_)
			self.losses.loc[len(self.losses)] = loss.flatten()
			print(f"iteration {_}/{self.max_iter}: loss: {loss}")
		self.visualize_training()
		self.save_model()

	def visualize_training(self):
		for column_name, column_values in self.losses.items():
			plt.plot(range(len(column_values)), column_values, label=column_name)
		plt.xlabel('Iteration')
		plt.ylabel('Loss')
		plt.title(f'Loss per Class over Iterations: batch size {self.batch}')
		plt.legend()
		plt.savefig('Loss_training.png')

	def predict(self, x):
		probabilities = self.softmax_(x)
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

		for row_name, row_data in df.iterrows():
			true_label = row_name
			for column_name, column_data in row_data.items():
				predicted_label = column_name
				df.at[true_label, predicted_label] = self.count(y, y_hat, true_label, predicted_label)

		if (label):
			res = df.loc[label, label]
		else:
			res = df
		print("the confusion matrix part:")
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

	def performance(self, y, y_hat, label):
		performance = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
		for l in label:
			param = self.parameteres(y, y_hat, l)
			
			acc = (param[0] + param[3] / (param[0] + param[1] + param[2] + param[3]))
			precision = 100 * (param[0] / (param[0] + param[2]))
			recall = 100 * (param[0] / (param[0] + param[1]))
			f1_score = (2 * precision * recall) / (1e-15 + precision + recall)

			performance['accuracy'].append(acc)
			performance['precision'].append(precision)
			performance['recall'].append(recall)
			performance['f1_score'].append(f1_score)

		return performance

	def save_model(self, filename='model.joblib'):
		model_data = {'thetas': self.theta.tolist(), 'label_encoder': self.label_encoder}
		joblib.dump(model_data, filename)

	def load_model(self, file_name='model.joblib'):
		model_data = joblib.load(file_name)
		self.theta = model_data['thetas']
		self.label_encoder = model_data['label_encoder']