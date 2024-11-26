from sklearn.datasets import load_iris
import utils.utils_math as mt
from sklearn import datasets
import pandas as pd
import numpy as np
import time
import sys

def hypothesis(thetas, data):
	dot_product = np.dot(thetas, data)
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
	
	r = learning_rate * (1 / X.shape[1]) * summ
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

def main():
	np.set_printoptions(suppress=True) # Suppress scientific notation
	iris = load_iris()

	# Convert it to a pandas DataFrame
	iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
	iris_df['species'] = iris.target
	iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
	# iris_df = iris_df.sample(frac=1, random_state=42).reset_index(drop=True)

	n_df = normalize(iris_df)

	n_df.insert(0, 'Bias', 1)
	thetas = np.array([0, 0, 0, 0, 0])
	X = n_df.iloc[:, :-1].values
	labels = ['setosa', 'versicolor', 'virginica']

	y = (n_df['species'] == 'setosa').astype(int).values
	start_time = time.time()
	thetas = gradient_descent(thetas, X, y)
	print(thetas)

	elapsed_time = time.time() - start_time
	print(f"Temps d'exécution : {elapsed_time:.6f} secondes")

if __name__ == "__main__":
    main()

# [-1.0007436  -0.20248627 -1.2148626   2.3112069  -2.11067198]
# python3 main.py  28.46s user 1.94s system 108% cpu 28.086 total

# [-11.7957484   -3.71207507   6.07681637 -10.96840883 -10.59296564]
# [-1.00074265 -0.20228006 -1.21493068  2.31103462 -2.11075178]
# [-20.04854867  -2.03749333  -2.8924317   16.56311666  13.84568525]

# import numpy as np
# import pandas as pd
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler
# from scipy.optimize import minimize

# # Load the Iris dataset
# iris = load_iris()
# X = iris.data
# y = (iris.target == 2).astype(int)  # Binary classification: 1 for versicolor, 0 for others

# # Normalize the features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Add intercept term to X
# X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add a column of ones for the intercept term

# # Define the sigmoid function
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))

# # Define the cost function
# def cost_function(theta, X, y):
#     m = len(y)
#     h = sigmoid(np.dot(X, theta))
#     return -1/m * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))

# # Define the gradient of the cost function
# def gradient(theta, X, y):
#     m = len(y)
#     h = sigmoid(np.dot(X, theta))
#     return 1/m * np.dot(X.T, (h - y))

# # Initialize theta
# initial_theta = np.zeros(X.shape[1])

# # Use a solver to minimize the cost function
# result = minimize(fun=cost_function, x0=initial_theta, args=(X, y), method='TNC', jac=gradient)

# # Get the optimized theta
# theta = result.x

# # Print the thetas
# print("Optimized Thetas for Versicolor vs All:")
# print(theta)