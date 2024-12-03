import utils.prepare_data as prep_d
import numpy as np

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

def hypothesis(thetas, X):
	z = X@thetas
	return 1 / (1 + np.exp(-z))

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
		print(f"{i},{labels[index]}")

def main():
	try:
		np.set_printoptions(suppress=True) # Suppress scientific notation
		df = prep_d.prepare_data('datasets/dataset_test.csv')
		df.insert(0, 'Bias', 1)

		X = df.select_dtypes(include=['number']).values
		r = []

		with open('logistic_thetas.txt', 'r') as file:
			for line in file:
				thetas = line.strip().split()
				thetas = list(map(float, thetas))
				r.append(list(map(float, line.strip().split())))

		predict(r, X, df)
	except Exception as e:
		print(f"{RED}Error: {e}{RESET}")

if __name__ == "__main__":
    main()