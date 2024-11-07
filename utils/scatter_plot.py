import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import utils_math as um
import mplcursors

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"  # Reset to default color

def display_correlation_graph(feature_1_y, feature_2_y):
	x = np.arange(1, len(feature_1_y) + 1)
	fig, ax = plt.subplots(figsize=(12, 8))
	plt.scatter(x, feature_1_y, color='b', s=70)
	scat = ax.scatter(x, feature_2_y, color='r', s=30)
	title = ax.text(0.5, 1.05, "", fontsize=12, transform=ax.transAxes, ha="center")

	def update(frame):
		if frame == 0:
			title.set_text("Initial Defense Against the Dark Arts values")
			new_y = feature_2_y
		elif frame == 1:
			new_y = feature_2_y * 100
			title.set_text("Defense Against the Dark Arts values * 100")
		else:
			new_y = feature_2_y * -100
			title.set_text("Defense Against the Dark Arts values to positive")
		scat.set_offsets(np.c_[x, new_y])  # Update positions of points
		return scat, title

	anim = animation.FuncAnimation(fig, update, frames=3, interval=2000)

	plt.text(0.05, 0.95, 'Blue: Astronomy score\nRed: Defense Against the Dark Arts', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
	plt.xlabel("X-axis label")
	plt.ylabel("Y-axis label")
	plt.grid(True)
	plt.show()

def correlation(data):
	if data.shape[1] != 2:
		return "ERROR"
		# raise ValueError("The data must have 2 columns")

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

def display_all_features_correlation_graph():
	values = [10, 20, 30, 40, 50]
	categories = ['A', 'B', 'C', 'D', 'E']
	tooltip_data = {
	'A': 'Additional info for bar A',
	'B': 'Additional info for bar B',
	'C': 'Additional info for bar C',
	'D': 'Additional info for bar D',
	'E': 'Additional info for bar E'
	}
	
	fig, ax = plt.subplots(figsize=(16, 9))
	ax.axhline(y="10", color='red', linestyle='-', linewidth=2)
	bars = ax.bar(categories, values)
	
	mplcursors.cursor(bars, hover=True).connect("add", lambda sel: sel.annotation.set_text(f'Value:'))

	ax.set_title('Sample Bar Chart')
	# plt.xlabel('Categories')
	# plt.ylabel('Values')

	# Display the chart
	plt.show()

def get_features_correlations(data):
	print(data.corr())
	print("--------------------------------------------------------------------")
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

		print(end="\n")
	
def main(): # Astronomy = Defense Against the Dark Arts * -100
	np.set_printoptions(suppress=True) # Suppress scientific notation
	dataset_test = "./datasets/dataset_test.csv"
	dataset_train = "./datasets/dataset_train.csv"

	data = pd.read_csv(dataset_train)

	feature_1_y = data.head(200)['Astronomy'].values
	feature_2_y = data.head(200)['Defense Against the Dark Arts'].values

	display_all_features_correlation_graph()

	# display_correlation_graph(feature_1_y, feature_2_y)
	feature_to_compare =	data.drop('Index', axis=1).drop('Hogwarts House', axis=1).drop('First Name', axis=1
							).drop('Last Name', axis=1).drop('Best Hand', axis=1).drop('Birthday', axis=1)
	get_features_correlations(feature_to_compare.head(10))

if __name__ == "__main__":
    main()