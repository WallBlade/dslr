# def histogram():
# 	pass

# def scatter_plot():
# 	pass

# def pair_plot():
# 	pass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Sample data for demonstration
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
scat = ax.scatter(x, y, color='b', s=100)  # Initial scatter plot with x and y
title = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center")  # Title placeholder

def update(frame):
	new_y = y + frame
	scat.set_offsets(np.c_[x, new_y])  # Update positions of points
	if frame == 0:
		title.set_text("Initial X values")
	elif frame == 1:
		title.set_text("X * 100")
	elif frame == 2:
		title.set_text("X to positive")
	# title.set_text(f"Frame: {frame}")

	return scat, title

# # Create the animation
ani = animation.FuncAnimation(fig, update, frames=3, interval=1000)

plt.xlabel("X-axis label")
plt.ylabel("Y-axis label")
plt.grid(True)
plt.show()