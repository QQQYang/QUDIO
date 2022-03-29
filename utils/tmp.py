import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))
line1, = ax.plot([1, 2, 3], label="Line 1", linestyle='--')
line2, = ax.plot([3, 2, 1], label="Line 2", linewidth=4)

# Create a legend for the first line.
first_legend = plt.legend(handles=[line1], loc='upper right')

# Add the legend manually to the current Axes.
plt.gca().add_artist(first_legend)

# Create another legend for the second line.
plt.legend(handles=[line2], loc='lower right')
ax.grid(True)
plt.tight_layout()

plt.show()