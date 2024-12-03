import utils.prepare_data as prep_d
import pandas as pd
import numpy as np
import time
import sys

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
		print(f"\033[32mIndex: {i} prediction: {labels[index]} prob: {prob[index]}\033[0m")

def main():
	# try:
	np.set_printoptions(suppress=True) # Suppress scientific notation
	df = prep_d.prepare_data('datasets/dataset_test.csv')
	df.insert(0, 'Bias', 1)

	X = df.select_dtypes(include=['number']).values
	with open('logistic_thetas.txt', 'r') as file:
		for line in file:
			# r.append(list(map(float, line.strip().split())))
			print(line)
	# r = [[-4.73492917, -1.30059946, -2.83179215,  2.80333461,  0.42985167,
    #     0.51813633,  3.25250019,  1.25574746, -1.17938403, -1.85469993,
    #     5.8323784 ], [-2.24272297,  2.07262412,  2.45016615,  1.02905415, -1.28709271,
    #    -1.60271137,  0.40124806, -0.17860478, -0.01654976, -0.67055768,
    #    -1.13140041], [-2.36776646, -1.16608572,  1.11252704, -0.18263422,  0.74618376,
    #     1.40804   , -0.53262887, -0.88297075,  0.0925333 ,  1.68819776,
    #    -0.94286206], [-4.59977224, -1.97150977, -0.34322703, -1.61553238, -0.17390371,
    #    -0.37221513, -0.09493884,  1.66789086,  0.58818215, -1.67648574,
    #    -0.41893518]]
	# predict(r, X, df)

if __name__ == "__main__":
    main()