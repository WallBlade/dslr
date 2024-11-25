import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris
import sys

def hypothesis(thetas, data):
	dot_product = thetas[0] * data[0] + thetas[1] * data[1] + thetas[2] * data[2] + thetas[3] * data[3] + thetas[4] * data[4]
	return 1 / (1 + np.exp(-dot_product))

def cost_function(thetas, X, y):
	j = 0

	for i in range(len(X)):
		h = hypothesis(thetas, X[i])
		j += y[i] * np.log(h) + (1 - y[i]) * np.log(1 - h)

	return (-1 / y.shape[0] * j)

def gradient(thetas, X, y):
	learning_rate = 0.1
	summ = 0

	for i in range(len(X)):
		h = hypothesis(thetas, X[i])
		summ += (h - y[i]) * X[i]
	
	r = learning_rate * 1 / y.shape[0] * summ
	return thetas - r

def gradient_descent(thetas, X, y):
	old_cost = cost_function(thetas, X, y)
	i = 0

	while True:
		thetas = gradient(thetas, X, y)
		new_cost = cost_function(thetas, X, y)
		print(f"New gradient: {thetas} new_cost: {new_cost} old_cost: {old_cost} at iteration {i}")
		if new_cost > old_cost or abs(new_cost - old_cost) < sys.float_info.epsilon:
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
	shuffled_df = iris_df.sample(frac=1, random_state=42).reset_index(drop=True)

	# Display the first few rows
	shuffled_df.insert(0, 'Bias', 1)
	thetas = np.array([0, 0, 0, 0, 0])
	X = shuffled_df.iloc[:, :-1].values
	y = (shuffled_df['species'] == 'versicolor').astype(int).values
	# thetas = gradient_descent(thetas, X, y)
	print(shuffled_df)

if __name__ == "__main__":
    main()