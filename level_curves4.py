import numpy as np
import matplotlib.pyplot as plt

# Define the foci points F1 and F2.
F1 = np.array([10, 10])
F2 = np.array([1, 2])

# The center of the ellipse is the midpoint of the foci.
c = (F1 + F2) / 2

# Define the semi-major axis length a.
# (Make sure that 2a is greater than the distance between the foci.)
# Here, 2a = 3, which is greater than the distance between F1 and F2 (which is 1).
foci_distance = np.linalg.norm(F2 - F1)
a = 25 #1.01*foci_distance / 2
if 2*a <= foci_distance:
    raise ValueError("2a must be greater than the distance between the foci.")

# Create a grid of x and y values over a 100x100 area.
# We'll center the grid on c and let it span from c[0]-50 to c[0]+50 (and similar for y).
grid_range = 50
x_vals = np.linspace(c[0] - grid_range, c[0] + grid_range, 400)
y_vals = np.linspace(c[1] - grid_range, c[1] + grid_range, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute the distances from each grid point to the foci.
D1 = (X - F1[0])**2 + (Y - F1[1])**2
D2 = (X - F2[0])**2 + (Y - F2[1])**2

# Define H such that the ellipse is given by H(x,y) = 0.
H = D1 + D2 - (2 * a)**2

# Plotting the ellipse.
plt.figure(figsize=(8, 6))

# Plot the H=0 level curve corresponding to the ellipse.
contour_zero = plt.contour(X, Y, H, levels=[0], colors='blue', linewidths=2)
plt.clabel(contour_zero, inline=True, fontsize=10)

# Optionally, plot a filled contour for additional context.
contours = plt.contourf(X, Y, H, levels=100, cmap='viridis', alpha=0.6)
plt.colorbar(contours)

# Plot the foci points and the center.
plt.plot(F1[0], F1[1], 'ro', label='F1')
plt.plot(F2[0], F2[1], 'ro', label='F2')
plt.plot(c[0], c[1], 'ko', label='Center')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Ellipse: $|PF_1| + |PF_2| = 2a$')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()


