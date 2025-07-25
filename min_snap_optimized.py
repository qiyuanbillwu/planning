from utils import a, Q_snap
import constants
import numpy as np
import matplotlib.pyplot as plt
import time

# Minimum snap cost function for x trajectory

DOF = 8  # Number of coefficients for the polynomial
n_segments = 2  # Number of segments in the trajectory

def Jx_snap(T1, T2):
    # Boundary conditions
    # start: hovering
    p0 = -1
    v0 = 0
    a0 = 0
    j0 = 0
    # end: hovering
    p2 = 1
    v2 = 0
    a2 = 0
    j2 = 0
    # gate states
    p1 = 0
    a1 = 0
    # v1 is continuous
    # j1 is continuous
    # Hessian matrices
    Q1 = Q_snap(T1)
    Q2 = Q_snap(T2)
    Q_total = np.block([
        [Q1, np.zeros_like(Q1)],
        [np.zeros_like(Q2), Q2]
    ])
    # Fixed derivatives
    df = np.array([[p0], [v0], [a0], [j0], [p2], [v2], [a2], [j2], [p1], [p1], [a1], [a1], [0], [0]])
    zeros_block = np.zeros((1, DOF))
    A = np.block([
        [a(0,0), zeros_block],
        [a(1,0), zeros_block],
        [a(2,0), zeros_block],
        [a(3,0), zeros_block],
        [zeros_block, a(0,T2)],
        [zeros_block, a(1,T2)],
        [zeros_block, a(2,T2)],
        [zeros_block, a(3,T2)],
        [a(0,T1), zeros_block],
        [zeros_block, a(0,0)],
        [a(2,T1), zeros_block],
        [zeros_block, a(2,0)],
        [a(1, T1), -a(1,0)],
        [a(3, T1), -a(3,0)],
        [a(1, T1), zeros_block],
        [a(3, T1), zeros_block]
    ])
    A_inv = np.linalg.inv(A)
    # print(A_inv)
    R = A_inv.T @ Q_total @ A_inv
    n_f = df.shape[0]
    n_p = DOF * n_segments - n_f
    R_FF = R[:n_f, :n_f]
    R_FP = R[:n_f, n_f:n_f+n_p]
    R_PP = R[n_f:n_f+n_p, n_f:n_f+n_p]
    dp_star = -np.linalg.inv(R_PP) @ R_FP.T @ df
    d = np.vstack([df, dp_star])
    px = A_inv @ d
    p1_coeffs = px[:DOF].flatten()
    p2_coeffs = px[DOF:].flatten()
    cost = p1_coeffs.T @ Q1 @ p1_coeffs + p2_coeffs.T @ Q2 @ p2_coeffs
    return cost, p1_coeffs, p2_coeffs

# Minimum snap cost function for y trajectory
def Jy_snap(T1, T2):
    # Boundary conditions 
    p0 = 1
    v0 = 0
    a0 = 0
    j0 = 0
    p2 = 1
    v2 = 0
    a2 = 0
    j2 = 0
    p1 = -1
    # v, a, j is continuous

    Q1 = Q_snap(T1)
    Q2 = Q_snap(T2)
    Q_total = np.block([
        [Q1, np.zeros_like(Q1)],
        [np.zeros_like(Q2), Q2]
    ])

    df = np.array([[p0], [v0], [a0], [j0], [p2], [v2], [a2], [j2], [p1], [p1], [0], [0], [0]])
    zeros_block = np.zeros((1, DOF))
    A = np.block([
        [a(0,0), zeros_block],
        [a(1,0), zeros_block],
        [a(2,0), zeros_block],
        [a(3,0), zeros_block],
        [zeros_block, a(0,T2)],
        [zeros_block, a(1,T2)],
        [zeros_block, a(2,T2)],
        [zeros_block, a(3,T2)],
        [a(0,T1), zeros_block],
        [zeros_block, a(0,0)],
        [a(1, T1), -a(1,0)],
        [a(2, T1), -a(2,0)],
        [a(3, T1), -a(3,0)],
        [a(1, T1), zeros_block],
        [a(2, T1), zeros_block],
        [a(3, T1), zeros_block]
    ])
    A_inv = np.linalg.inv(A)
    R = A_inv.T @ Q_total @ A_inv
    n_f = df.shape[0]
    n_p = DOF * n_segments - n_f
    R_FF = R[:n_f, :n_f]
    R_FP = R[:n_f, n_f:n_f+n_p]
    R_PP = R[n_f:n_f+n_p, n_f:n_f+n_p]
    dp_star = -np.linalg.inv(R_PP) @ R_FP.T @ df
    d = np.vstack([df, dp_star])
    py = A_inv @ d
    p1_coeffs = py[:DOF].flatten()
    p2_coeffs = py[DOF:].flatten()
    cost = p1_coeffs.T @ Q1 @ p1_coeffs + p2_coeffs.T @ Q2 @ p2_coeffs
    return cost, p1_coeffs, p2_coeffs

# Minimum snap cost function for z trajectory
def Jz_snap(T1, T2):
    constants.T1 = T1
    constants.T2 = T2
    # Boundary conditions 
    p0 = 0.5
    v0 = 0
    a0 = 0
    j0 = 0
    p2 = 0.5
    v2 = 0
    a2 = 0
    j2 = 0
    p1 = 2
    a1 = -constants.g
    # v, j is continuous

    Q1 = Q_snap(T1)
    Q2 = Q_snap(T2)
    Q_total = np.block([
        [Q1, np.zeros_like(Q1)],
        [np.zeros_like(Q2), Q2]
    ])
    df = np.array([[p0], [v0], [a0], [j0], [p2], [v2], [a2], [j2], [p1], [p1], [a1], [a1], [0], [0]])
    zeros_block = np.zeros((1, DOF))
    A = np.block([
        [a(0,0), zeros_block],
        [a(1,0), zeros_block],
        [a(2,0), zeros_block],
        [a(3,0), zeros_block],
        [zeros_block, a(0,T2)],
        [zeros_block, a(1,T2)],
        [zeros_block, a(2,T2)],
        [zeros_block, a(3,T2)],
        [a(0,T1), zeros_block],
        [zeros_block, a(0,0)],
        [a(2,T1), zeros_block],
        [zeros_block, a(2,0)],
        [a(1, T1), -a(1,0)],
        [a(3, T1), -a(3,0)],
        [a(1, T1), zeros_block],
        [a(3, T1), zeros_block],
    ])
    A_inv = np.linalg.inv(A)
    R = A_inv.T @ Q_total @ A_inv
    n_f = df.shape[0]
    n_p = DOF * n_segments - n_f
    R_FF = R[:n_f, :n_f]
    R_FP = R[:n_f, n_f:n_f+n_p]
    R_PP = R[n_f:n_f+n_p, n_f:n_f+n_p]
    dp_star = -np.linalg.inv(R_PP) @ R_FP.T @ df
    d = np.vstack([df, dp_star])
    pz = A_inv @ d
    p1_coeffs = pz[:DOF].flatten()
    p2_coeffs = pz[DOF:].flatten()
    cost = p1_coeffs.T @ Q1 @ p1_coeffs + p2_coeffs.T @ Q2 @ p2_coeffs
    return cost, p1_coeffs, p2_coeffs

def J_snap(T1, T2, k_T = 10000):
    cost_x, _, _ = Jx_snap(T1, T2)
    cost_y, _, _ = Jy_snap(T1, T2)
    cost_z, _, _ = Jz_snap(T1, T2)
    return cost_x + cost_y + cost_z + k_T * (T1 + T2)

# Gradient descent parameters
learning_rate = 1e-6
max_iterations = 2000
tolerance = 1e-4

T1_current = 1.6
T2_current = 1.6
T1_history = [T1_current]
T2_history = [T2_current]
cost_history = []

print("Starting minimum snap gradient descent optimization...")
start_time_gd = time.time()
for iteration in range(max_iterations):
    current_cost = J_snap(T1_current, T2_current)
    cost_history.append(current_cost)
    delta = 1e-8
    grad_T1 = (J_snap(T1_current + delta, T2_current) - current_cost) / delta
    grad_T2 = (J_snap(T1_current, T2_current + delta) - current_cost) / delta
    T1_new = T1_current - learning_rate * grad_T1
    T2_new = T2_current - learning_rate * grad_T2
    T1_new = max(0.1, float(T1_new))
    T2_new = max(0.1, float(T2_new))
    T1_change = abs(T1_new - T1_current)
    T2_change = abs(T2_new - T2_current)
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: T1={T1_current:.6f}, T2={T2_current:.6f}, Cost={current_cost:.6f}")
    if T1_change < tolerance and T2_change < tolerance:
        print(f"Converged after {iteration} iterations")
        break
    T1_current = T1_new
    T2_current = T2_new
    T1_history.append(float(T1_current))
    T2_history.append(float(T2_current))
    cost_history.append(float(current_cost))
end_time_gd = time.time()
gradient_descent_time = end_time_gd - start_time_gd
print(f"Gradient descent completed in {gradient_descent_time:.6f} seconds")
print(f"Optimal T1: {T1_current:.6f}")
print(f"Optimal T2: {T2_current:.6f}")

print(f"Final cost: {current_cost:.6f}")

# override coefficients

# T1_current = 1.5
# T2_current = 1.5

# Save optimized time parameters
optimized_time = {
    "T1": float(T1_current),
    "T2": float(T2_current)

}

# Save coefficients for x, y, and z trajectories
_, x_p1, x_p2 = Jx_snap(T1_current, T2_current)
_, y_p1, y_p2 = Jy_snap(T1_current, T2_current)
_, z_p1, z_p2 = Jz_snap(T1_current, T2_current)

coefficients = {
    "p1_coeffs": {
        "x": x_p1.tolist(),
        "y": y_p1.tolist(),
        "z": z_p1.tolist()
    },
    "p2_coeffs": {
        "x": x_p2.tolist(),
        "y": y_p2.tolist(),
        "z": z_p2.tolist()
    }
}

import json
with open("data/min_snap_coeffs.json", "w") as f:
    json.dump(coefficients, f)
with open("data/min_snap_time.json", "w") as f:
    json.dump(optimized_time, f)

# Plot optimization history
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(T1_history)
plt.title('T1 Optimization History')
plt.xlabel('Iteration')
plt.ylabel('T1')
plt.grid(True)
plt.subplot(1, 3, 2)
plt.plot(T2_history)
plt.title('T2 Optimization History')
plt.xlabel('Iteration')
plt.ylabel('T2')
plt.grid(True)
plt.subplot(1, 3, 3)
plt.plot(cost_history)
plt.title('Cost Function History')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.grid(True)
plt.tight_layout()
plt.show()
