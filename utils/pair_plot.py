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

def display_KDE_graph(ax, data, feature_name):
	ravenclaw = data[data['Hogwarts House'] == 'Ravenclaw'].iloc[:, 1].to_numpy()
	slytherin = data[data['Hogwarts House'] == 'Slytherin'].iloc[:, 1].to_numpy()
	gryffindor = data[data['Hogwarts House'] == 'Gryffindor'].iloc[:, 1].to_numpy()
	hufflepuff = data[data['Hogwarts House'] == 'Hufflepuff'].iloc[:, 1].to_numpy()

	sns.kdeplot(ravenclaw, fill=True, color="magenta", legend=False, ax=ax)
	sns.kdeplot(slytherin, fill=True, color="cyan", legend=False, ax=ax)
	sns.kdeplot(gryffindor, fill=True, color="black", legend=False, ax=ax)
	sns.kdeplot(hufflepuff, fill=True, color="orange", legend=False, ax=ax)
	
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	# ax.set_yticklabels([])
	# ax.set_xticklabels([])
	# ax.set_xlabel(feature_name[:4])
	# ax.set_ylabel(feature_name[:4])


def display_pair_graph(ax, data):
	# Check that contains: House, Feature1, Feature2
	if len(data.columns) != 3:
		return
	data.dropna()
	x_values = data.iloc[:, 1]
	y_values = data.iloc[:, 2]
	colors = get_colors(data['Hogwarts House'])
	ax.scatter(x_values, y_values, c=colors, alpha=0.5, s=1)

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
			
			if feature_x == feature_y:
				display_KDE_graph(axs[y, x], data[['Hogwarts House', feature_x]], feature_x)
			else:
				display_pair_graph(axs[y, x], data[['Hogwarts House', feature_x, feature_y]])

			if x == 0: axs[y, x].set_ylabel(feature_y[:4])
			else : axs[y, x].set_ylabel('')

			# else:
			# 	axs[y, x].set_yticklabels([])
			# if y != 11:
				# axs[y, x].set_xticklabels([])  # Hide x-axis tick labels	
			axs[y, x].set_xticklabels([])  # Hide x-axis tick labels	
			axs[y, x].set_yticklabels([])
			axs[y, x].set_xlabel(feature_x[:4])
			x += 1
		x = 0
		y += 1

	plt.tight_layout()
	plt.subplots_adjust(hspace=0.1, wspace=0.1)
	
	manager = plt.get_current_fig_manager()
	manager.full_screen_toggle()  # Toggle fullscreen
	plt.show()
	# print("--------------------------------------------------------------------")
	# sns.pairplot(data, hue="Hogwarts House")
	# plt.show()

if __name__ == "__main__":
	main()