import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ------------------------------
# Simulation Parameters
# ------------------------------
dt = 0.01  # Time step (s), 100 Hz
num_steps = 250  # Number of simulation steps

# Control input reference
u_reference = -2.0  # Desired control input

# Initial state: [position, velocity, u_hat]
initial_position = 2.0  # meters
initial_velocity = -1  # m/s
initial_u_hat = 0.0  # initial estimate of control input

# Controller and Barrier Function parameters
kappa = 5.0
alpha_1 = 5.0  # Gain for alpha(h)
alpha_2 = 5.0  # Gain for alpha(h)
h_alpha = 5.0  # Gain for h_alpha
u_max = 0.5  # Maximum allowable control input

# ------------------------------
# Data Storage for Plots
# ------------------------------
time_data = []
h_1_data = []
h_2_data = []
u_reference_data = []
u_safe_data = []
softmin_data = []
mu_data = []  # Control input μ
b1_data = []
b2_data = []

# ------------------------------
# System Dynamics Function
# ------------------------------
def system_dynamics(state, mu):
    """
    Computes the time derivative of the state.

    Parameters:
        state (np.array): Current state [position, velocity, u_hat]
        mu (float): Control input

    Returns:
        np.array: Derivative of the state
    """
    position, velocity, u_hat = state
    d_position = velocity
    d_velocity = u_hat
    d_u_hat = -u_hat + mu
    return np.array([d_position, d_velocity, d_u_hat])

# ------------------------------
# RK4 Integration Step Function
# ------------------------------
def rk4_step(state, mu, dt, dynamics_func):
    """
    Performs one RK4 integration step.

    Parameters:
        state (np.array): Current state
        mu (float): Control input
        dt (float): Time step
        dynamics_func (function): Function to compute state derivatives

    Returns:
        np.array: Updated state after one RK4 step
    """
    k1 = dynamics_func(state, mu)
    k2 = dynamics_func(state + 0.5 * dt * k1, mu)
    k3 = dynamics_func(state + 0.5 * dt * k2, mu)
    k4 = dynamics_func(state + dt * k3, mu)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ------------------------------
# Initialize State
# ------------------------------
state = np.array([initial_position, initial_velocity, initial_u_hat])

# ------------------------------
# Simulation Loop
# ------------------------------
for step in range(num_steps):
    time_elapsed = step * dt

    # Desired control input is fixed to u_reference
    u_hat_desired = u_reference

    # Safety Filter (Control Barrier Function)
    # Extract state variables
    position, velocity, u_hat = state

    # Define barrier functions
    b_1 = position
    b_2 = velocity + alpha_1 * position

    # Compute h₁ and h₂
    h_1 = u_hat + (alpha_1 + alpha_2) * velocity + alpha_1 * alpha_2 * position
    h_2 = u_max - u_hat

    # Lie derivatives for h₁
    Lg_h_1 = 1.0
    Lf_h_1 = (alpha_1 + alpha_2 - 1.0) * u_hat + alpha_1 * alpha_2 * velocity

    # Lie derivatives for h₂
    Lg_h_2 = -1.0
    Lf_h_2 = u_hat

    # Composite CBF using softmin
    h_composite = -np.log(np.exp(-kappa * h_1) + np.exp(-kappa * h_2)) / kappa
    softmin_data.append(h_composite)

    # Compute Lie derivatives for composite CBF
    exp_term1 = np.exp(-kappa * (h_1 - h_composite))
    exp_term2 = np.exp(-kappa * (h_2 - h_composite))
    Lf_h_composite = exp_term1 * Lf_h_1 + exp_term2 * Lf_h_2
    Lg_h_composite = exp_term1 * Lg_h_1 + exp_term2 * Lg_h_2

    # Simple controller (desired control input adjustment)
    mu_desired = u_hat + 10.0 * (u_hat_desired - u_hat)

    # Set up and solve QP
    mu = cp.Variable()
    objective = cp.Minimize((mu - mu_desired) ** 2)  # Minimize deviation from desired control input
    cbf_constraint = Lg_h_composite * mu + Lf_h_composite + h_alpha * h_composite >= 0
    constraints = [cbf_constraint]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status != cp.OPTIMAL:
        print(f"QP Infeasible at time: {time_elapsed:.2f}s. Using fallback control input.")
        mu_value = mu_desired  # Use a safe fallback control input
    else:
        mu_value = mu.value

    # Store control input μ for plotting
    mu_data.append(mu_value)

    # Apply RK4 to update the state
    state = rk4_step(state, mu_value, dt, system_dynamics)

    # Store data for plots
    time_data.append(time_elapsed)
    h_1_data.append(h_1)  # h₁ (Position-based CBF)
    h_2_data.append(h_2)  # h₂ (u_max - u)
    u_reference_data.append(u_hat_desired)
    u_safe_data.append(state[2])  # u (renamed from u_hat)
    b1_data.append(b_1)
    b2_data.append(b_2)

# ------------------------------
# Plotting Section
# ------------------------------

# Set global font sizes
plt.rcParams.update({
    'font.size': 18,          # Default text size
    'axes.titlesize': 22,     # Axes title size
    'axes.labelsize': 20,     # Axes label size
    'legend.fontsize': 18,    # Legend font size
    'xtick.labelsize': 16,    # X-axis tick label size
    'ytick.labelsize': 16,    # Y-axis tick label size
})

# ------------------------------
# Figure 1: μ, u, h₁, h₂, and h_composite
# ------------------------------
fig1, axs1 = plt.subplots(4, 1, figsize=(12, 18), sharex=True)

# Subplot 1: Control Input μ
axs1[0].plot(time_data, mu_data, label=r"$\mu$", color="orange")
axs1[0].set_ylabel(r"$\mu$")
axs1[0].legend(loc="upper right")
axs1[0].grid(True)

# Subplot 2: Control Input u
axs1[1].plot(time_data, u_safe_data, label=r"$u$", color="purple")
# Make a red line along the y-axis at u_max
axs1[1].axhline(y=u_max, color="red", linestyle="--", label=r"$u_{\max}$")
axs1[1].set_ylabel("u")
axs1[1].legend(loc="upper right")
axs1[1].grid(True)

# Subplot 3: Barrier Functions h₁, h₂, and h_composite
# Set a maximum value for h₁ and h₂ for better visualization
axs1[2].plot(time_data, h_1_data, label=r"$h_1$", color="blue")
axs1[2].plot(time_data, h_2_data, label=r"$h_2$", color="green")
#axs1[2].plot(time_data, softmin_data, label="h_composite (Softmin)", color="red")
axs1[2].set_ylim(min(min(h_1_data), min(h_2_data)) - 0.1, max(h_2_data) + 2)
axs1[2].legend(loc="upper right")
axs1[2].grid(True)

# Subplot position
axs1[3].plot(time_data, b1_data, label=r'$p$', color="magenta")
# Make a red line along the y-axis for p_min=0
axs1[3].axhline(y=0, color="red", linestyle="--", label=r"$p_{\min}$")
axs1[2].set_xlabel("Time (s)")
axs1[3].grid(True)
axs1[3].legend(loc="upper right")

plt.tight_layout()
fig1.savefig("mu_u_h_functions_plot.pdf")
plt.close()

# ------------------------------
# Figure 2: b₁ Over Time
# ------------------------------
plt.figure(figsize=(12, 4))
plt.plot(time_data, b1_data, label="b₁", color="magenta")
plt.title("Position Over Time")
plt.xlabel("Time (s)")
plt.ylabel("p")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig("b1_plot.pdf")
plt.close()

# ------------------------------
# Figure 3: b₂ Over Time
# ------------------------------
plt.figure(figsize=(12, 4))
plt.plot(time_data, b2_data, label="b₂", color="cyan")
plt.title("b₂ Over Time")
plt.xlabel("Time (s)")
plt.ylabel("b₂")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig("b2_plot.pdf")
plt.close()

# ------------------------------
# Optional: Display Plots
# ------------------------------
# Uncomment the following line if you wish to display the plots interactively.
# plt.show()






