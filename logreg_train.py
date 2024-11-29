import utils.utils_math as mt
import utils.prepare_data as prep_d
import pandas as pd
import numpy as np
import time
import sys

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

def hypothesis(thetas, X):
	z = X@thetas
	return 1 / (1 + np.exp(-z))

def cost_function(thetas, X, y):
	j = np.sum(y * np.log(hypothesis(thetas, X)) + (1 - y) * np.log(1 - hypothesis(thetas, X)))
	return -1 / y.shape[0] * j

def create_mini_batches(X, y, batch_size):
    n_samples = X.shape[0]
    
    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Shuffle the data
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Split into mini-batches
    mini_batches = []
    for i in range(0, n_samples, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, y_batch))
    
    return mini_batches

def gradient(thetas, X, y):
	m = X.shape[0]
	sum = (1 / m) * X.T @ (hypothesis(thetas, X) - y)
	return sum

def gradient_descent(X, y):
	thetas = np.zeros(11)
	old_cost = float('inf')
	learning_rate = 0.5

	for i in range(2000):
		# new_cost = 0
		mini_batches = create_mini_batches(X, y, 32)
		for X_mini, y_mini in mini_batches:
			grad = gradient(thetas, X_mini, y_mini)
			thetas = thetas - learning_rate * grad
			# new_cost += cost_function(thetas, X_mini, y_mini)
		# if new_cost > old_cost or abs(old_cost - new_cost) < sys.float_info.epsilon:
		# 	print(f"TROUVÉ at iteration {i} old_cost: {old_cost} new_cost: {new_cost}")
		# 	return thetas
		# old_cost = new_cost

	# for i in range(100000):
	# 	grad = gradient(thetas, X, y)
	# 	thetas = thetas - learning_rate * grad
	# 	# new_cost = cost_function(thetas, X, y)
	# 	# print(f"New gradient: {thetas} new_cost: {new_cost} old_cost: {old_cost} at iteration {i}")

	# 	# if new_cost > old_cost or abs(old_cost - new_cost) < sys.float_info.epsilon:
	# 	# 	print(f"TROUVÉ at iteration {i} old_cost: {old_cost} new_cost: {new_cost}")
	# 	# 	break
	
	# 	# old_cost = new_cost
	# 	# i += 1

	return thetas

def predict(thetas, X, df):
	labels = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
	errors = 0

	for i, row in df.iterrows():
		prob = np.array([
			hypothesis(thetas[0], X[i]) * 100, # Gryffindor
			hypothesis(thetas[1], X[i]) * 100, # Hufflepuff
			hypothesis(thetas[2], X[i]) * 100, # Ravenclaw
			hypothesis(thetas[3], X[i]) * 100 # Slytherin
		])
		index = np.argmax(prob)
		if labels[index] == row['Hogwarts House']:
			print(f"\033[32mindex: {i} specie: {row['Hogwarts House']} prediction: {labels[index]} prob: {prob[index]}\033[0m")
		else:
			errors += 1
			print(f"\033[31mindex: {i} specie: {row['Hogwarts House']} prediction: {labels[index]} prob: {prob[index]}\033[0m")
	
	print(f"Errors: {errors} / {len(df)}")

def main():
	np.set_printoptions(suppress=True) # Suppress scientific notation
	df = prep_d.prepare_data('datasets/dataset_train.csv')
	df.insert(0, 'Bias', 1)


	X = df.select_dtypes(include=['number']).values
	r = []
	start_time = time.time()

	for label in ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']:
		y = (df['Hogwarts House'] == label).astype(int).values
		thetas = gradient_descent(X, y)
		r.append(thetas)
		print(f"Thetas {label}: {thetas}")
	
	elapsed_time = time.time() - start_time
	print(f"Temps d'exécution : {elapsed_time:.6f} secondes")
	# predict(r, X, df)

if __name__ == "__main__":
    main()