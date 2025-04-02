import numpy as np
import matplotlib.pyplot as plt

# First define the points
l_1 = np.array([100, 100])
l_2 = np.array([1, 1])

# Define the center and the difference vector
c = (l_1 + l_2) / 2
s = l_2 - l_1

# Define a position (unused in the sqrt computation, kept for consistency)
x = np.array([0, 0])
p = np.linalg.norm(s) / 6  # Originally p was defined as norm(s)/6, now using /2

# r = x - c transformation: compute the rotated matrix R_hat and the normalized R
R_hat = np.array([[s[0], -s[1]], [s[1], s[0]]])
R = R_hat / np.linalg.norm(s)
Lambda = np.array([[0.1, 0], [0, 10]])

# Compute the effective matrix A = R * Lambda * R^T
A = R @ Lambda @ R.T

# Create a 2D grid of x and y values covering a 500x500 region centered at c.
# The grid spans from c[0]-250 to c[0]+250 and c[1]-250 to c[1]+250.
x_vals = np.linspace(c[0] - 250, c[0] + 250, 500)
y_vals = np.linspace(c[1] - 250, c[1] + 250, 500)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute r = [x - c[0], y - c[1]] for each point in the grid
R1 = X - c[0]
R2 = Y - c[1]

# Calculate h(x,y) = r^T A r - p^2
# Expanding: A[0,0]*(R1)^2 + 2*A[0,1]*R1*R2 + A[1,1]*(R2)^2 - p^2
H = A[0, 0]*R1**2 + 2*A[0, 1]*R1*R2 + A[1, 1]*R2**2 - p**2

# Compute sqrt(h(x,y)) only where h(x,y) is non-negative.
H_sqrt = np.where(H >= 0, np.sqrt(H), np.nan)

# Plotting the level curves of sqrt(h(x,y))
plt.figure(figsize=(10, 8))

# First, plot the h=0 level curve (note that sqrt(0)=0, so this is unchanged)
contour_zero = plt.contour(X, Y, H_sqrt, levels=[0], colors='blue', linewidths=2)
plt.clabel(contour_zero, inline=True, fontsize=10)

# Optionally, plot a filled contour map of sqrt(h(x,y)) for additional context
contours = plt.contourf(X, Y, H_sqrt, levels=100, cmap='viridis', alpha=0.6)
plt.colorbar(contours)

# Plot the points l_1, l_2 and the center c for reference
plt.plot(l_1[0], l_1[1], 'ro', label='l_1')
plt.plot(l_2[0], l_2[1], 'ro', label='l_2')
plt.plot(c[0], c[1], 'ko', label='Center')

plt.xlabel('x')
plt.ylabel('y')
plt.title(r'Level Curves of $\sqrt{h(x,y)}$, where $h(x,y)=r^T A r - p^2$')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()



