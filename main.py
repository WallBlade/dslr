import numpy as np
import pandas as pd

def main():
	np.set_printoptions(suppress=True) # Suppress scientific notation
	dataset_test = "./datasets/dataset_test.csv"
	dataset_train = "./datasets/dataset_train.csv"
      
	data = pd.read_csv(dataset_train)

	print(data)

if __name__ == "__main__":
    main()