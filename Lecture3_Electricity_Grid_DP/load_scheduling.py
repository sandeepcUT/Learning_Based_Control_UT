"""
Load Scheduling with Slew Rate Constraints
ECE Control Theory Demo

We optimize electricity generation to match a demand forecast,
while respecting slew-rate (ramp-up/ramp-down) constraints.

Based on: "Task-based End-to-end Model Learning in Stochastic Optimization"
(Donti et al., NeurIPS 2017)
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def run_load_scheduling(demand_timeseries, slew_penalty, x0=3):
    """
    Solve the load scheduling problem with given slew constraint.

    Args:
        demand_timeseries (np.ndarray): Demand forecast (length H).
        slew_penalty (float): Max allowed change in generation per time step.
        x0 (float): Initial generation value.

    Returns:
        (solution, cost)
        solution: np.ndarray of generated amounts.
        cost: optimal objective value.
    """
    H = len(demand_timeseries)
    n = 1  # state dimension

    # Decision variable: generation at each step
    x = cp.Variable((n, H + 1))

    # Objective and constraints
    cost = 0
    constr = [x[:, 0] == x0]

    for t in range(H):
        # Penalize deviation from demand
        cost += cp.sum_squares(x[:, t] - demand_timeseries[t])

        # Slew-rate constraint: limit change in generation
        constr += [cp.norm(x[:, t + 1] - x[:, t], 1) <= slew_penalty]

    # Solve convex optimization
    problem = cp.Problem(cp.Minimize(cost), constr)
    obj = problem.solve()

    return x.value.flatten(), obj


if __name__ == "__main__":
    # Demand forecast (toy example)
    demand = np.array([1, 1, 3, 3, 5, 5, 5, 2, 2, 3, 3, 1])

    # Compare strict vs relaxed slew constraints
    slew_values = [0.5, 1.5, 2.5]

    plt.figure(figsize=(8, 5))

    for slew in slew_values:
        gen, cost = run_load_scheduling(demand, slew_penalty=slew, x0=3)
        plt.plot(gen, label=f"Generated (slew={slew}, cost={cost:.2f})")

    # Plot demand curve
    plt.plot(demand, label="Demand", color="black", linestyle="--", linewidth=2)

    plt.xlabel("Time step")
    plt.ylabel("Power generated")
    plt.title("Load Scheduling with Slew Rate Constraints")
    plt.legend()
    plt.grid(True)
    plt.show()