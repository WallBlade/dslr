import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

def main(): # Astronomy = Defense Against the Dark Arts * -100
	np.set_printoptions(suppress=True) # Suppress scientific notation
	dataset_test = "./datasets/dataset_test.csv"
	dataset_train = "./datasets/dataset_train.csv"

	data = pd.read_csv(dataset_train)

	feature_1_y = data.head(10)['Astronomy'].values
	feature_2_y = data.head(10)['Defense Against the Dark Arts'].values
	x = np.arange(1, len(feature_1_y) + 1)

	print(feature_1_y)
	print(feature_2_y)

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

	animation.FuncAnimation(fig, update, frames=3, interval=2000)

	plt.text(0.05, 0.95, 'Blue: Astronomy score\nRed: Defense Against the Dark Arts', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
	plt.xlabel("X-axis label")
	plt.ylabel("Y-axis label")
	plt.grid(True)
	plt.show()

if __name__ == "__main__":
    main()