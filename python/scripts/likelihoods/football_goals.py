import matplotlib.pyplot as plt
import numpy as np

# Weight matrix 
A = np.array([[2.0,1.5,1.0],
              [1.5,0.7,0.5],
              [1,0.5,0.4]])

# Probabilities
P = np.exp(A)
Z = np.sum(P)
P = P / Z

fig, ax = plt.subplots()
im = ax.imshow(P, cmap="viridis")

# Set tick positions and labels
num_rows, num_cols = P.shape
ax.set_xticks(np.arange(num_cols))
ax.set_yticks(np.arange(num_rows))
ax.set_xticklabels(np.arange(num_cols))
ax.set_yticklabels(np.arange(num_rows))

# Move x-axis ticks to the top
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

# Annotate each cell with the corresponding value of P
for i in range(num_rows):
    for j in range(num_cols):
        text = f"{P[i, j]:.2f}"  # Format with 2 decimal places
        ax.text(j, i, text, ha="center", va="center", color="black", fontsize=12, fontweight="bold")

# Add colorbar
plt.colorbar(im, label="Probability")

plt.show()
