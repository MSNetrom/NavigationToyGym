import numpy as np
import matplotlib.pyplot as plt

# First define the points

l_1 = np.array([1, 1])
l_2 = np.array([1, 2])


# Define the center and diff
c = (l_1 + l_2) / 2
s = l_2 - l_1

# Define position
x = np.array([0, 0])
p = np.linalg.norm(s) / 2 #/ 6

# r = x - c

R_hat = np.array([[s[0], -s[1]], [s[1], s[0]]])
R = R_hat / np.linalg.norm(s)
Lambda = np.array([[1, 0], [0, 1]])

# Compute the effective matrix A = R_hat * Lambda * R_hat^T
A = R @ Lambda @ R.T

# Create a 2D grid of x and y values. 
# The limits are chosen relative to the center c.
x_vals = np.linspace(c[0] - 2, c[0] + 2, 400)
y_vals = np.linspace(c[1] - 2, c[1] + 2, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute r = x - c for each point in the grid
R1 = X - c[0]
R2 = Y - c[1]

# Calculate h = r^T * A * r - p^2
# This expands to: A[0,0]*(R1)^2 + 2*A[0,1]*R1*R2 + A[1,1]*(R2)^2 - p^2
H = A[0,0]*R1**2 + 2*A[0,1]*R1*R2 + A[1,1]*R2**2 - p**2

# Plotting the level curves
plt.figure(figsize=(8, 6))

# First, plot the h=0 level curve
contour_zero = plt.contour(X, Y, H, levels=[0], colors='blue', linewidths=2)
plt.clabel(contour_zero, inline=True, fontsize=10)

# Optionally, plot a filled contour map of H for additional context
contours = plt.contourf(X, Y, H, levels=100, cmap='viridis', alpha=0.6)
plt.colorbar(contours)

# Plot the points l_1, l_2 and the center c for reference
plt.plot(l_1[0], l_1[1], 'ro', label='l_1')
plt.plot(l_2[0], l_2[1], 'ro', label='l_2')
plt.plot(c[0], c[1], 'ko', label='Center')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Level Curves of $h(x,y)=r^T A r - p^2$')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()



