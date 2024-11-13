import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main(): # Astronomy = Defense Against the Dark Arts * -100
	np.set_printoptions(suppress=True) # Suppress scientific notation
	data = pd.read_csv("datasets/dataset_train.csv")
	data = data.drop('Index', axis=1)
	data = data.drop('First Name', axis=1)
	data = data.drop('Last Name', axis=1)
	# data = data.drop('Best Hand', axis=1) # Convert to numerical value
	# data = data.drop('Birthday', axis=1) # Convert to numerical value
    
	# data = data[['Hogwarts House', 'Astronomy']]
	# print(data.head)

	# print(data)
	# data = sns.load_dataset("iris") # Built-in dataset in seaborn
	print(data)
	print("--------------------------------------------------------------------")
	# Pair plot with hue based on species
	sns.pairplot(data, hue="Hogwarts House")
	plt.show()

if __name__ == "__main__":
    main()