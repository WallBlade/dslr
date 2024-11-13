import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import utils_math as um
import mplcursors

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

def correlation(data) -> float:
	"""
	Returns correlations coefficient between two features using Pearson correlation.
	The closer number is from 1 or -1, the stronger the correlation is.

	parameters: data (pandas.DataFrame): The data to calculate the correlation.
	returns: float from -1 to 1
	"""

	if data.shape[1] != 2:
		raise ValueError("The dataFrame must have 2 columns")

	data = data.dropna()
	f1 = data.iloc[:, 0:1].values
	f2 = data.iloc[:, 1:2].values

	f1_mean = um.mean_val(f1)
	f2_mean = um.mean_val(f2)

	sum1 = 0
	sum2 = 0
	sum3 = 0

	for i in range(len(f1)):
		if np.isnan(f1[i]) or np.isnan(f2[i]):
			continue
		sum1 += (f1[i][0] - f1_mean) * (f2[i][0] - f2_mean)

	for num in f1:
		num = num[0]
		if np.isnan(num):
			continue
		sum2 += (num - f1_mean) ** 2

	for num in f2:
		num = num[0]
		if np.isnan(num):
			continue
		sum3 += (num - f2_mean) ** 2

	numerator = sum1
	denominator = np.sqrt(sum2 * sum3)
	correlation = np.round(numerator / denominator, 2)

	return correlation

def get_correlation_coefficients(data):
	"""
	This function calculates the correlation between all the features of a dataset, prints them and returns them in array.

	parameters: data (pandas.DataFrame): The data containing all the features to calculate the correlations.
	returns: list of dictionaries {feature_1, feature_2, correlation}
	"""
	results = []
	print("\t", end="")
	for feature in data:
		print(feature[:4], end=".\t")
	print(end="\n")

	for i in range(data.shape[1]):
		f_1 = data.columns[i]
		print(f"{f_1[:4]}:", end="\t")
		for y in range(data.shape[1]):
			f_2 = data.columns[y]
			if y <= i:
				print("/", end="\t")
				continue
	
			corr = correlation(data[[f_1, f_2]])
			if corr == 1 or corr == -1:
				print(f"{RED}1.0{RESET}", end="\t")
			elif corr > 0.8 or corr < -0.8:
				print(f"{YELLOW}{corr}{RESET}", end="\t")
			else:
				print(f"{corr}", end="\t")
			results.append({'feature_1': f_1, 'feature_2': f_2, 'correlation': corr})

		print(end="\n")
	return results

def display_correlation_coefficients(ax1, correlations):
	"""
	Display all the correlations coefficients.

	parameters: ax1 (axes object): first graph, correlations (list of dictionaries): The correlations to display.
	returns: matplotlib.animation.FuncAnimation
	"""
	y1 = np.array([item["correlation"] for item in correlations])
	x1 = np.arange(0, len(correlations))

	colors = ["red" if value <= -1 else "red" if value >= 1 else "orange" if value <= -0.8 else "orange" if value >= 0.80 else "green" for value in y1]
	bars = ax1.bar(x1, y1, color=colors)
	mplcursors.cursor(bars, hover=True).connect("add", lambda sel: sel.annotation.set_text(f"Correlation between {correlations[sel.index]['feature_1']}\nand {correlations[sel.index]['feature_2']} is {correlations[sel.index]['correlation']}"))
	
	ax1.set_title('Pearson correlation coefficient (-1 and 1 means a linear relationship)')
	ax1.get_xaxis().set_visible(False)
	ax1.set_ylabel('Pearson correlation coefficient')

def display_feature_comparision(fig, ax2, data):
	"""
	Display the correlation between two features using a scatter plot and animation.

	parameters: fig (figure object): Graph container holder, ax2: second graph, data (pandas.DataFrame): The data to calculate the correlation.
	returns: matplotlib.animation.FuncAnimation
	"""

	feature_1_y = data['Astronomy'].values
	feature_2_y = data['Defense Against the Dark Arts'].values
	x2 = np.arange(1, len(feature_1_y) + 1)
	feature_1_scatter = ax2.scatter(x2, feature_1_y, color='b', s=30, label='Astronomy grade')
	feature_2_scatter = ax2.scatter(x2, feature_2_y, color='r', s=10, label='Defense Against the Dark Arts grade')
	ratio = feature_1_y[0] / feature_2_y[0]
	legend = ax2.legend(loc='upper left')

	def update(frame):
		if frame == 0:
			legend.get_texts()[1].set_text(f"Defense Against the Dark Arts grade * -100")
			new_y = feature_2_y * ratio
		elif frame == 1:
			legend.get_texts()[1].set_text("Defense Against the Dark Arts grade")
			new_y = feature_2_y
		feature_2_scatter.set_offsets(np.c_[x2, new_y])
		return feature_2_scatter, legend

	anim = animation.FuncAnimation(fig, update, frames=2, interval=2000)

	ax2.set_title("Astronomy and Defense Against the Dark Arts grades comparison")
	ax2.get_xaxis().set_visible(False)
	ax2.set_ylabel("Grade")
	return anim

def main():
	# ---- Set up the dataset ---- #
	np.set_printoptions(suppress=True)
	dataset_test = "./datasets/dataset_test.csv"
	dataset_train = "./datasets/dataset_train.csv"
	data = pd.read_csv(dataset_train)
	data = data.dropna()
	
	# ---- Get correlations ---- #
	feature_to_compare = data.loc[:, 'Arithmancy':'Flying']
	correlations = get_correlation_coefficients(feature_to_compare)

	# ---- Display graphs ---- #
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(17, 9))
	display_correlation_coefficients(ax1, correlations)
	anim = display_feature_comparision(fig, ax2, data)
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
    main()