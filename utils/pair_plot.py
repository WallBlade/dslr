import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_colors(houses):
	colors = []
	for house in houses:
		if house == 'Ravenclaw':
			colors.append('magenta')
		elif house == 'Slytherin':
			colors.append('cyan')
		elif house == 'Gryffindor':
			colors.append('black')
		elif house == 'Hufflepuff':
			colors.append('orange')
	return colors

def display_pair_graph(ax, data):
	# Check that contains: House, Feature1, Feature2
	if len(data.columns) != 3:
		return
	# print(data.columns[1], data.columns[2])
	data.dropna()
	x = data.iloc[:, 1]
	y = data.iloc[:, 2]
	colors = get_colors(data['Hogwarts House'])
	ax.scatter(x, y, c=colors, alpha=0.5, s=1)
	# ax.get_xaxis().set_visible(False)
	# ax.get_yaxis().set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)

def main():
	np.set_printoptions(suppress=True)
	data = pd.read_csv("datasets/dataset_train.csv")
	data = data.drop('Defense Against the Dark Arts', axis=1)

	fig, axs = plt.subplots(12, 12, figsize=(16, 10))
	y = 0
	x = 0

	for feature_y in data.loc[:, 'Arithmancy':'Flying']:
		for feature_x in data.loc[:, 'Arithmancy':'Flying']:
			# if x == 0:
			# 	axs[y, x].set_xlabel(feature_x[:4])
			# if y == 0:
			# 	axs[y, x].set_ylabel(feature_y[:4])

			display_pair_graph(axs[y, x], data[['Hogwarts House', feature_x, feature_y]])
			x += 1
		x = 0
		y += 1

	axs[1, 0].set_xlabel('TEST1')
	axs[1, 1].set_xlabel('TEST2')
	plt.tight_layout()
	plt.subplots_adjust(hspace=0.1, wspace=0.1)

	plt.show()
	print("--------------------------------------------------------------------")
	# sns.pairplot(data, hue="Hogwarts House")
	# plt.show()

if __name__ == "__main__":
    main()