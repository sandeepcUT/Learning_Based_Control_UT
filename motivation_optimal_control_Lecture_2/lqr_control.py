
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# System parameters
n = 2  # state dimension
m = 1  # control dimension
T = 80  # time horizon
delta_t = 0.1  # time interval
mass = 1  # mass

# Dynamics matrices
A = np.array([[1, delta_t], [0, 1]])
B = np.array([[0], [delta_t/mass]])

# Initial condition
x_0 = [5, -2]

# Cost function scalars
traj_cost_scalar = 10.0
control_scalar = 1.0

# List of max control norms and corresponding colors for plotting
norm_u_max_list = [0.4, 0.5, 1.0, 5.0]
color_list = ['r', 'g', 'b', 'black']

# Plot setup
plt.figure(figsize=(10, 8))

# Loop through different max control norms
for norm_u, col in zip(norm_u_max_list, color_list):
    # Define variables for state and control
    x = cp.Variable((n, T + 1))
    u = cp.Variable((m, T))

    # Initialize cost and constraints
    cost = 0
    constraints = []

    for t in range(T):
        cost += traj_cost_scalar * cp.sum_squares(x[:, t + 1]) + control_scalar * cp.sum_squares(u[:, t])
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t], cp.norm(u[:, t], "inf") <= norm_u]

    # Final constraints
    constraints += [x[:, T] == 0, x[:, 0] == x_0]

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    # Plotting results
    plt.subplot(3, 1, 1)
    plt.plot(u[0, :].value, color=col, label=f'||u_t|| <= {norm_u}')
    plt.ylabel("u_t", fontsize=14)
    plt.xticks([])

    plt.subplot(3, 1, 2)
    plt.plot(x[0, :].value, color=col, label=f'||u_t|| <= {norm_u}')
    plt.ylabel("x_t (position)", fontsize=14)
    plt.xticks([])

    plt.subplot(3, 1, 3)
    plt.plot(x[1, :].value, color=col, label=f'||u_t|| <= {norm_u}')
    plt.ylabel("v_t (velocity)", fontsize=14)
    plt.xticks([])

plt.legend()
plt.show()
