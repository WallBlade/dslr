from sklearn.datasets import load_iris
import utils.utils_math as mt
from sklearn import datasets
import pandas as pd
import numpy as np
import time
import sys

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
	return thetas - sum * 0.1
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

def gradient_descent(thetas, X, y, num_iters):
	old_cost = cost_function(thetas, X, y)
	i = 0
	learning_rate = 0.1
	# v = np.zeros_like(thetas)

	while True:
		thetas = gradient(thetas, X, y)
		# v = 0.9 * v + (1 - 0.9) * grad
		# thetas = thetas - learning_rate * v
		# thetas = thetas - learning_rate * grad

		new_cost = cost_function(thetas, X, y)
		# print(f"New gradient: {thetas} new_cost: {new_cost} old_cost: {old_cost} at iteration {i}")

		if new_cost > old_cost or abs(new_cost - old_cost) < sys.float_info.epsilon:
			print("TROUVe a l'iteration", i)
			break
	
		old_cost = new_cost
		i += 1
	return thetas

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

	y = (n_df['species'] == 'versicolor').astype(int).values
	start_time = time.time()
	thetas = gradient_descent(thetas, X, y, 10000000)
	print(thetas)

# New gradient: [-7.19387122 -2.41904796  4.33654502 -6.9522864  -6.49201985] at iteration 199999

	elapsed_time = time.time() - start_time
	print(f"Temps d'exécution : {elapsed_time:.6f} secondes")

if __name__ == "__main__":
    main()

# [-11.7957484   -3.71207507   6.07681637 -10.96840883 -10.59296564]
# [-1.00074265 -0.20228006 -1.21493068  2.31103462 -2.11075178]
# [-20.04854867  -2.03749333  -2.8924317   16.56311666  13.84568525]