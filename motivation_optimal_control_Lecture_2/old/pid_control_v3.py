
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as lg

# System parameters
m = 1  # mass
tol = 1e-3  # tolerance for eigenvalue check
delta_t = 0.05  # time interval
max_steps = 500  # maximum time steps
n = 2  # state dimension

# Dynamics matrices
A = np.array([[0, 1], [0, 0]])
B = np.array([[0], [1/m]])

# Function to run the PID simulation
def run_pid_simulation(kp_list=None, kd_list=None, initial_condition=None, varying_param="kp"):
    color_list = ['b', 'g', 'r', 'c']
    plt.figure(figsize=(10, 8))
    
    for idx, param in enumerate(kp_list if varying_param == "kp" else kd_list):
        kp = param if varying_param == "kp" else 1.0
        kd = param if varying_param == "kd" else 1.0
        K = np.array([kp, kd])
        
        # Closed loop dynamics
        A_closed_loop = np.array([[0, 1], [-kp/m, -kd/m]])
        Eigenvalues, Eigenvectors = lg.eig(A_closed_loop)
        
        if np.real(Eigenvalues[0]) >= tol or np.real(Eigenvalues[1]) >= tol:
            raise ValueError("Eigenvalues must have negative real parts.")
        
        Lambda = np.diag(Eigenvalues)
        T = Eigenvectors
        Tinv = np.linalg.inv(Eigenvectors)
        
        s_vec = [initial_condition[0, 0]]
        v_vec = [initial_condition[1, 0]]
        u_vec = []
        
        for k in range(max_steps):
            t = delta_t * k
            matrix_exponential = np.array([[np.exp(Eigenvalues[0]*t), 0], [0, np.exp(Eigenvalues[1]*t)]])
            x_t = T.dot(matrix_exponential).dot(Tinv).dot(initial_condition)
            u = -K.dot(x_t)
            
            s_vec.append(np.real(x_t[0, 0]))
            v_vec.append(np.real(x_t[1, 0]))
            u_vec.append(np.real(u[0]))
        
        label_str = f"$k_{{{'p' if varying_param == 'kp' else 'd'}}} = {param}$"
        
        plt.subplot(3, 1, 1)
        plt.plot(s_vec, label=label_str, color=color_list[idx % len(color_list)])
        plt.ylabel("$s_t$", fontsize=16)
        plt.xticks([])
        
        plt.subplot(3, 1, 2)
        plt.plot(v_vec, label=label_str, color=color_list[idx % len(color_list)])
        plt.ylabel("$v_t$", fontsize=16)
        plt.xticks([])
        
        plt.subplot(3, 1, 3)
        plt.plot(u_vec, label=label_str, color=color_list[idx % len(color_list)])
        plt.ylabel("$u_t$", fontsize=16)
        plt.xticks([])
    
    plt.legend(loc='right')
    plt.show()

# Initial conditions
initial_condition_kp = np.array([[10.0], [0.0]])
initial_condition_kd = np.array([[8.0], [0.0]])

# Varying kp with fixed kd = 1
kp_list = [0.2, 0.5, 2, 5, 10]
run_pid_simulation(kp_list=kp_list, initial_condition=initial_condition_kp, varying_param="kp")

# Varying kd with fixed kp = 1
kd_list = [0.5, 1, 2.1, 5, 10]
run_pid_simulation(kd_list=kd_list, initial_condition=initial_condition_kd, varying_param="kd")
