import pygame
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Parameters
dt = 0.01  # Time step, 100 Hz

# Control input u
acceleration = 1.2  # Control acceleration (m/s^2)
# State x
initial_position, initial_velocity, initial_u_hat = 0.6, 0.0, 0.0  # Initial states

kappa = 5
p = 6.0
alpha_1 = 27  # Gain for alpha(h)
alpha_2 = 10  # Gain for alpha(h)
h_alpha = 58  # Gain for h_alpha
u_max = 53

# Data lists for plots
time_data = []
h_x_data = []
u_hat_desired_data = []
u_safe_data = []
mu_data = []  # For storing mu values

# Pygame setup
pygame.init()
width, height = 800, 1000  # Width: 800 pixels, Height: 700 pixels
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("1D Double Integrator with Safety Filter (CBF)")
clock = pygame.time.Clock()

# Constants for visual representation
center_x = width // 2
scale = 100  # Scaling factor to convert meters to pixels

# Matplotlib setup for real-time plotting
# Set figsize to (8, 3.5) inches and dpi=100 to match 800x350 pixels
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 5), dpi=100)
fig.subplots_adjust(hspace=0.6)  # Space between plots
canvas = FigureCanvas(fig)

# Plot settings
ax1.set_title("Constraint Value h(x) over Time")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("h(x) = Position")
ax1.axhline(0, color="red", linestyle="--")  # Safety boundary line
ax1.grid(True)

ax2.set_title("Reference and Safe Acceleration over Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Acceleration (m/s²)")
ax2.legend(["Desired Acceleration", "Safe Acceleration"], loc="upper right")
ax2.grid(True)

ax3.set_title("Control Input μ over Time")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("μ")
ax3.legend(["Control Input μ"], loc="upper right")
ax3.grid(True)

# Initialize state
state = np.array([initial_position, initial_velocity, initial_u_hat])

# System dynamics function
def system_dynamics(state, mu):
    position, velocity, u_hat = state
    d_position = velocity
    d_velocity = u_hat
    d_u_hat = -u_hat + mu
    return np.array([d_position, d_velocity, d_u_hat])

# RK4 step function
def rk4_step(state, mu, dt, dynamics_func):
    k1 = dynamics_func(state, mu)
    k2 = dynamics_func(state + 0.5 * dt * k1, mu)
    k3 = dynamics_func(state + 0.5 * dt * k2, mu)
    k4 = dynamics_func(state + dt * k3, mu)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Run the simulation
running = True
time_elapsed = 0.0
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Read keyboard input
    keys = pygame.key.get_pressed()
    u_hat_desired = 0  # Reset desired control input
    if keys[pygame.K_LEFT]:
        u_hat_desired = -acceleration
    elif keys[pygame.K_RIGHT]:
        u_hat_desired = acceleration

    # Safety Filter (Control Barrier Function)
    # Define state x
    position, velocity, u_hat = state

    # Barrier function h(x) = position
    h_1 = u_hat + (alpha_1 + alpha_2)*velocity + alpha_1*alpha_2*position
    Lg_h_1 = 1
    Lf_h_1 = (alpha_1 + alpha_2 -1)*u_hat + alpha_1*alpha_2*velocity

    # Input constraints
    h_2 = u_max - u_hat
    Lg_h_2 = -1
    Lf_h_2 = u_hat

    # Define composite CBF
    h_composite = -np.log(np.exp(-kappa*h_1) + np.exp(-kappa*h_2)) / kappa

    # Compute its Lie derivatives
    exp_term1 = np.exp(-kappa*(h_1 - h_composite))
    exp_term2 = np.exp(-kappa*(h_2 - h_composite))
    Lf_h_composite = exp_term1 * Lf_h_1 + exp_term2 * Lf_h_2
    Lg_h_composite = exp_term1 * Lg_h_1 + exp_term2 * Lg_h_2

    # Simple controller
    mu_desired = u_hat + 10*(u_hat_desired - u_hat)  # Adjusted controller

    # Set up and solve QP
    mu = cp.Variable()
    objective = cp.Minimize((mu - mu_desired) ** 2)  # Minimize deviation from desired control input
    cbf_constraint = Lg_h_composite * mu + Lf_h_composite + h_alpha * h_composite >= 0
    constraints = [cbf_constraint]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status != cp.OPTIMAL:
        print("QP Infeasible at time:", time_elapsed)
        mu_value = mu_desired  # Use a safe fallback control input
    else:
        mu_value = mu.value

    # Store mu for plotting
    mu_data.append(mu_value)

    # Apply RK4 to update the state
    state = rk4_step(state, mu_value, dt, system_dynamics)

    # Store data for plots
    time_data.append(time_elapsed)
    h_x_data.append(state[0])  # position
    u_hat_desired_data.append(u_hat_desired)
    u_safe_data.append(state[2])  # u_hat
    time_elapsed += dt

    # Clear screen and draw Pygame elements
    screen.fill((255, 255, 255))  # White background
    pos_pixel = int(center_x + state[0] * scale)
    pygame.draw.line(screen, (255, 0, 0), (center_x, 0), (center_x, height // 2), 2)  # Draw red constraint line at x=0
    pygame.draw.circle(screen, (0, 0, 255), (pos_pixel, height // 4), 10)  # Draw particle in upper half of screen

    # Update Matplotlib plots
    ax1.clear()
    ax1.plot(time_data, h_x_data, label="h(x) = Position", color="blue")
    ax1.axhline(0, color="red", linestyle="--")  # Safety boundary line
    ax1.set_title("Constraint Value h(x) over Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("h(x) = Position")
    ax1.legend()
    ax1.grid(True)

    ax2.clear()
    ax2.plot(time_data, u_hat_desired_data, label="Desired Acceleration", color="green")
    ax2.plot(time_data, u_safe_data, label="Safe Acceleration", color="purple")
    ax2.set_title("Reference and Safe Acceleration over Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Acceleration (m/s²)")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    ax3.clear()
    ax3.plot(time_data, mu_data, label="Control Input μ", color="orange")
    ax3.set_title("Control Input μ over Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("μ")
    ax3.legend(loc="upper right")
    ax3.grid(True)

    # Render the updated Matplotlib figure to a pygame-compatible image
    canvas.draw()
    plot_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    plot_image = plot_image.reshape(canvas.get_width_height()[::-1] + (3,))
    plot_surface = pygame.surfarray.make_surface(plot_image.swapaxes(0, 1))  # Adjust for pygame's coordinate system

    # Blit the plot in the lower half of the pygame window
    screen.blit(plot_surface, (0, height // 2))  # Position the plot at the bottom half of the window

    pygame.display.flip()
    clock.tick(100)  # Run at 100 Hz

# After exiting the loop, save the plots
fig.savefig("simulation_plots.png")
pygame.quit()



