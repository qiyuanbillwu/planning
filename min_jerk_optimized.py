from utils import a5, Q5
import constants
import numpy as np
import matplotlib.pyplot as plt
import json
import time

def Jx(T1, T2):

    # Define the boundary conditions
    # initial position, velocity, acceleration
    p0 = -1
    v0 = 0
    a0 = 0

    # final position, velocity, acceleration
    p2 = 1
    v2 = 0
    a2 = 0

    # gate states
    # leave the velocity unspecified
    p1 = 0
    a1 = 0

    # construct the total Hessian matrix
    # start_time = time.time()
    Q1 = Q5(T1)
    Q2 = Q5(T2)
    Q_total = np.block([
        [Q1, np.zeros_like(Q1)],
        [np.zeros_like(Q2), Q2]
    ])

    # define the fixed/specified derivatives
    # fix the position at the gate
    # leave velocity, acceleration unspecified
    df = np.array([[p0], [v0], [a0], [p2], [v2], [a2], [p1], [p1], [a1], [a1], [0]])

    # construct the A matrix
    zeros_block = np.zeros((1, 6))
    A = np.block([
        [a5(0,0), zeros_block], # p0
        [a5(1,0), zeros_block], # v0
        [a5(2,0), zeros_block], # a0
        [zeros_block, a5(0,T2)], # p2
        [zeros_block, a5(1,T2)], # v2
        [zeros_block, a5(2,T2)], # a2
        [a5(0,T1), zeros_block], # p1 (end of seg 1)
        [zeros_block, a5(0,0)], # p1 (start of seg 2)
        [a5(2,T1), zeros_block], # a1 (end of seg 1)
        [zeros_block, a5(2,0)], # a1 (start of seg 2)
        [a5(1, T1), -a5(1,0)], # v1 
        [a5(1, T1), zeros_block], # v1
    ])

    # Compute the inverse of A
    A_inv = np.linalg.inv(A)

    R = A_inv.T @ Q_total @ A_inv

    # Partition R based on the sizes of df and dp
    n_f = df.shape[0]  # number of fixed variables
    n_p = 6 * 2 - n_f  # number of free variables

    R_FF = R[:n_f, :n_f]
    R_FP = R[:n_f, n_f:n_f+n_p]
    R_PF = R[n_f:n_f+n_p, :n_f]
    R_PP = R[n_f:n_f+n_p, n_f:n_f+n_p]

    dp_star = -np.linalg.inv(R_PP) @ R_FP.T @ df

    # Assemble the complete solution vector d from df and dp_star
    d = np.vstack([df, dp_star])

    px = A_inv @ d

    # Extract coefficients for each polynomial
    p1_coeffs = px[:6].flatten()  # First 6 coefficients for polynomial 1
    p2_coeffs = px[6:].flatten()  # Last 6 coefficients for polynomial 2

    return p1_coeffs.T @ Q1 @ p1_coeffs + p2_coeffs.T @ Q2 @ p2_coeffs 

def Jy(T1, T2):
    """
    Cost function for y trajectory optimization using minimum jerk
    """
    
    # Define the boundary conditions for y trajectory
    # initial position, velocity, acceleration
    p0 = 1
    v0 = 0
    a0 = 0

    # final position, velocity, acceleration
    p2 = 1
    v2 = 0
    a2 = 0

    # gate states
    p1 = -1  # from min_jerk_y: p1 = -1
    v1 = 0

    # construct the total Hessian matrix
    Q1 = Q5(T1)
    Q2 = Q5(T2)
    Q_total = np.block([
        [Q1, np.zeros_like(Q1)],
        [np.zeros_like(Q2), Q2]
    ])

    # define the fixed/specified derivatives
    df = np.array([[p0], [v0], [a0], [p2], [v2], [a2], [p1], [p1], [v1], [v1], [0]])

    # construct the A matrix
    zeros_block = np.zeros((1, 6))
    A = np.block([
        [a5(0,0), zeros_block], # p0
        [a5(1,0), zeros_block], # v0
        [a5(2,0), zeros_block], # a0
        [zeros_block, a5(0,T2)], # p2
        [zeros_block, a5(1,T2)], # v2
        [zeros_block, a5(2,T2)], # a2
        [a5(0,T1), zeros_block], # p1 (end of seg 1)
        [zeros_block, a5(0,0)], # p1 (start of seg 2)
        [a5(2,T1), zeros_block], # a1 (end of seg 1)
        [zeros_block, a5(2,0)], # a1 (start of seg 2)
        [a5(1, T1), -a5(1,0)], # v1 
        [a5(1, T1), zeros_block], # v1
    ])

    # Compute the inverse of A
    A_inv = np.linalg.inv(A)

    R = A_inv.T @ Q_total @ A_inv

    # Partition R based on the sizes of df and dp
    n_f = df.shape[0]  # number of fixed variables
    n_p = 6 * 2 - n_f  # number of free variables

    R_FF = R[:n_f, :n_f]
    R_FP = R[:n_f, n_f:n_f+n_p]
    R_PF = R[n_f:n_f+n_p, :n_f]
    R_PP = R[n_f:n_f+n_p, n_f:n_f+n_p]

    dp_star = -np.linalg.inv(R_PP) @ R_FP.T @ df

    # Assemble the complete solution vector d from df and dp_star
    d = np.vstack([df, dp_star])

    px = A_inv @ d

    # Extract coefficients for each polynomial
    p1_coeffs = px[:6].flatten()  # First 6 coefficients for polynomial 1
    p2_coeffs = px[6:].flatten()  # Last 6 coefficients for polynomial 2

    return p1_coeffs.T @ Q1 @ p1_coeffs + p2_coeffs.T @ Q2 @ p2_coeffs 

def Jz(T1, T2):
    """
    Cost function for z trajectory optimization using minimum jerk
    """
    # Update constants for this iteration
    constants.T1 = T1
    constants.T2 = T2
    
    # Define the boundary conditions for z trajectory
    # initial position, velocity, acceleration
    p0 = 0.5
    v0 = 0
    a0 = 0

    # final position, velocity, acceleration
    p2 = 0.5
    v2 = 0
    a2 = 0

    # gate states
    p1 = 2
    v1 = 0
    a1 = -constants.g

    # construct the total Hessian matrix
    Q1 = Q5(T1)
    Q2 = Q5(T2)
    Q_total = np.block([
        [Q1, np.zeros_like(Q1)],
        [np.zeros_like(Q2), Q2]
    ])

    # define the fixed/specified derivatives
    df = np.array([[p0], [v0], [a0], [p2], [v2], [a2], [p1], [p1], [v1], [v1], [a1], [a1]])

    # construct the A matrix
    zeros_block = np.zeros((1, 6))
    A = np.block([
        [a5(0,0), zeros_block], # p0
        [a5(1,0), zeros_block], # v0
        [a5(2,0), zeros_block], # a0
        [zeros_block, a5(0,T2)], # p2
        [zeros_block, a5(1,T2)], # v2
        [zeros_block, a5(2,T2)], # a2
        [a5(0,T1), zeros_block], # p1 (end of seg 1)
        [zeros_block, a5(0,0)], # p1 (start of seg 2)
        [a5(1,T1), zeros_block], # v1 (end of seg 1)
        [zeros_block, a5(1,0)], # v1 (start of seg 2)
        [a5(2, T1), zeros_block], # a1 (end of seg 1)
        [zeros_block, a5(2,0)], # a1 (start of seg 2)
    ])

    # Compute the inverse of A
    A_inv = np.linalg.inv(A)

    R = A_inv.T @ Q_total @ A_inv

    # Partition R based on the sizes of df and dp
    n_f = df.shape[0]  # number of fixed variables
    n_p = 6 * 2 - n_f  # number of free variables

    R_FF = R[:n_f, :n_f]
    R_FP = R[:n_f, n_f:n_f+n_p]
    R_PF = R[n_f:n_f+n_p, :n_f]
    R_PP = R[n_f:n_f+n_p, n_f:n_f+n_p]

    dp_star = -np.linalg.inv(R_PP) @ R_FP.T @ df

    # Assemble the complete solution vector d from df and dp_star
    d = np.vstack([df, dp_star])

    px = A_inv @ d

    # Extract coefficients for each polynomial
    p1_coeffs = px[:6].flatten()  # First 6 coefficients for polynomial 1
    p2_coeffs = px[6:].flatten()  # Last 6 coefficients for polynomial 2

    return p1_coeffs.T @ Q1 @ p1_coeffs + p2_coeffs.T @ Q2 @ p2_coeffs 

k_T = 10000

def J(T1, T2):
    return Jx(T1, T2) + Jy(T1, T2) + Jz(T1, T2) + k_T * (T1 + T2)

# Start Generation Here

# Gradient descent parameters
learning_rate = 1e-5
max_iterations = 1000
tolerance = 1e-3

# Initial guess for T1 and T2
T1_current = 1.0
T2_current = 2.0

# Store history for plotting
T1_history = [T1_current]
T2_history = [T2_current]
cost_history = []

print("Starting gradient descent optimization...")

# Start timing
start_time_gd = time.time()

for iteration in range(max_iterations):
    # Calculate current cost
    current_cost = J(T1_current, T2_current)
    cost_history.append(current_cost)
    
    # Calculate gradients using finite differences
    delta = 1e-8
    grad_T1 = (J(T1_current + delta, T2_current) - current_cost) / delta
    grad_T2 = (J(T1_current, T2_current + delta) - current_cost) / delta
    
    # Update T1 and T2
    T1_new = T1_current - learning_rate * grad_T1
    T2_new = T2_current - learning_rate * grad_T2
    
    # Ensure T1 and T2 remain positive
    T1_new = max(0.1, float(T1_new))
    T2_new = max(0.1, float(T2_new))
    
    # Check convergence
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

# End timing
end_time_gd = time.time()
gradient_descent_time = end_time_gd - start_time_gd

print(f"Gradient descent completed in {gradient_descent_time:.6f} seconds")
print(f"Optimal T1: {T1_current:.6f}")
print(f"Optimal T2: {T2_current:.6f}")
print(f"Final cost: {current_cost:.6f}")

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

# End Generation Here


