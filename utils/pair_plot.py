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
			colors.append('yellow')
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
	ax.scatter(x, y, c=colors, alpha=0.5)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

def main():
	np.set_printoptions(suppress=True)
	data = pd.read_csv("datasets/dataset_train.csv")
	data = data.drop('Defense Against the Dark Arts', axis=1)
	# data = data.loc[:, 'Arithmancy':'Flying']

	fig, axs = plt.subplots(12, 12, figsize=(16, 10))
	i = 0
	y = 0

	# display_pair_graph(axs[i, y], data[['Hogwarts House', 'Arithmancy', 'Astronomy']])
	for feature_y in data.loc[:, 'Arithmancy':'Flying']:
		for feature_x in data.loc[:, 'Arithmancy':'Flying']:
			display_pair_graph(axs[i, y], data[['Hogwarts House', feature_y, feature_x]])
			y += 1
		y = 0
		i += 1
	# 		print(f"{feature_y}/{feature_x}")
	plt.tight_layout()
	plt.axis('off')
	for i in range(12):
		axs[i, 0].get_yaxis().set_visible(True)
		axs[i, 0].set_ylabel('X-axis')
	# axs[1, 0].set_ylabel('Y-axis')
	plt.show()
	print("--------------------------------------------------------------------")
	# sns.pairplot(data, hue="Hogwarts House")
	# plt.show()

if __name__ == "__main__":
    main()