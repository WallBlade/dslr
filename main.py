from sklearn.datasets import load_iris
import utils.utils_math as mt
from sklearn import datasets
import pandas as pd
import numpy as np
import time
import sys

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

def z_score(col, mean, std):
    return[((x - mean) / std) if pd.notna(x) else None for x in col]

def	normalize(df):
	numeric_cols = df.select_dtypes(include=['number'])
	categorical_cols = df.select_dtypes(include=['object'])
	normalized_df = pd.DataFrame()

	for column in numeric_cols:
		column_data = df[column]

		mean = mt.mean_val(column_data)
		std = mt.std_val(column_data)

		normalized_column = z_score(column_data, mean, std)
		normalized_df[column] = normalized_column
    
	normalized_df = pd.concat([normalized_df, categorical_cols], axis=1)
	return normalized_df

def hypothesis(thetas, X):
	z = X@thetas
	return 1 / (1 + np.exp(-z))

def cost_function(thetas, X, y):
	j = np.sum(y * np.log(hypothesis(thetas, X)) + (1 - y) * np.log(1 - hypothesis(thetas, X)))
	return -1 / y.shape[0] * j
	# j = 0

	# for i in range(len(X)):
	# 	h = hypothesis(thetas, X[i])
	# 	j += y[i] * np.log(h) + (1 - y[i]) * np.log(1 - h)

	# return (-1 / y.shape[0] * j)

def gradient(thetas, X, y):
	m = X.shape[0]
	sum = (1 / m) * X.T @ (hypothesis(thetas, X) - y)
	return sum
	# print(thetas - learning_rate * sum)
	# print(hypothesis(thetas, X) - y)
	# print("-----------------")
	# print(X)
	# sum = sum / m
	# r = learning_rate * sum
	# return thetas - r
	# sum = 0

	# for i in range(len(X)):
	# 	h = hypothesis(thetas, X[i])
	# 	sum += (h - y[i]) * X[i]
	
	# sum = sum / m

	# r = learning_rate * sum
	# return thetas - r

def gradient_descent(thetas, X, y):
	old_cost = cost_function(thetas, X, y)
	i = 0
	learning_rate = 0.5
	v = np.zeros_like(thetas)

	for i in range(100000):
		grad = gradient(thetas, X, y)
		thetas = thetas - learning_rate * grad
		# v = 0.9 * v + (1 - 0.9) * grad
		# thetas = thetas - learning_rate * v

		new_cost = cost_function(thetas, X, y)
		# print(f"New gradient: {thetas} new_cost: {new_cost} old_cost: {old_cost} at iteration {i}")

		if new_cost > old_cost or abs(new_cost - old_cost) < sys.float_info.epsilon:
			print("TROUVe a l'iteration", i)
			break
	
		old_cost = new_cost
		i += 1
	return thetas

def predict(thetas, X, df):
	labels = ['setosa', 'versicolor', 'virginica']
	for i, row in df.iterrows():
		prob = np.array([
			hypothesis(thetas[0], X[i]) * 100,
			hypothesis(thetas[1], X[i]) * 100,
			hypothesis(thetas[2], X[i]) * 100
		])
		index = np.argmax(prob)
		if labels[index] == row['species']:
			print(f"\033[32mspecie: {row['species']} prediction: {labels[index]} prob: {prob[index]}\033[0m")
		else:
			print(f"\033[31mspecie: {row['species']} prediction: {labels[index]} prob: {prob[index]}\033[0m")


def main():
	np.set_printoptions(suppress=True) # Suppress scientific notation
	iris = load_iris()

	# Convert it to a pandas DataFrame
	iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
	iris_df['species'] = iris.target
	iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

	n_df = normalize(iris_df)
	# n_df = iris_df
	n_df.insert(0, 'Bias', 1)
	
	thetas = np.array([0, 0, 0, 0, 0])
	X = n_df.iloc[:, :-1].values
	labels = ['setosa', 'versicolor', 'virginica']
	r = []
	for label in labels:
		y = (n_df['species'] == label).astype(int).values
		# start_time = time.time()
		thetas = gradient_descent(thetas, X, y)
		r.append(thetas)
		print(thetas)
		# elapsed_time = time.time() - start_time
		# print(f"Temps d'exécution : {elapsed_time:.6f} secondes")
	predict(r, X, n_df)

if __name__ == "__main__":
    main()

# [-11.7957484   -3.71207507   6.07681637 -10.96840883 -10.59296564]
# [-1.00074265 -0.20228006 -1.21493068  2.31103462 -2.11075178]
# [-20.04854867  -2.03749333  -2.8924317   16.56311666  13.84568525]