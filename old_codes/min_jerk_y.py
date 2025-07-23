from utils import a5, Q_jerk
from constants import T1, T2
import numpy as np
import matplotlib.pyplot as plt
import json

# Option to save coefficients to file
save_coeffs_to_file = True  # Set to False to skip saving

# Threshold for small coefficients
threshold = 1e-10

# Define the boundary conditions
# initial position, velocity, acceleration
p0 = 1
v0 = 0
a0 = 0

# final position, velocity, acceleration
p2 = 1
v2 = 0
a2 = 0

# gate states
# leave acceleration unspecified
p1 = -1
v1 = 0

# construct the total Hessian matrix
Q1 = Q5(T1)
Q2 = Q5(T2)
Q_total = np.block([
    [Q1, np.zeros_like(Q1)],
    [np.zeros_like(Q2), Q2]
])

# define the fixed/specified derivatives
# fix the position at the gate
# leave velocity, acceleration unspecified
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
    [a5(1,T1), zeros_block], # v1 (end of seg 1)
    [zeros_block, a5(1,0)], # v1 (start of seg 2)
    [a5(2, T1), -a5(2,0)], # a1 
    [a5(2, T1), zeros_block], # a1
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

print_dp_star = False  # Set this to False to suppress printing

if print_dp_star:
    print("Optimal free variables (dp_star):")
    labels = [
        "v1 (velocity at x1)", 
        "a1 (acceleration at x1)"
    ]
    for i, val in enumerate(dp_star.flatten()):
        if i < len(labels):
            print(f"{labels[i]}: {val}")
        else:
            print(f"dp_star[{i}]: {val}")

# Assemble the complete solution vector d from df and dp_star
d = np.vstack([df, dp_star])

px = A_inv @ d

print_px = False  # Set this to False to suppress printing

if print_px:
    print("Solution vector px (all polynomial coefficients):")

# Extract coefficients for each polynomial
p1_coeffs = px[:6].flatten()  # First 6 coefficients for polynomial 1
p2_coeffs = px[6:].flatten()  # Last 6 coefficients for polynomial 2

if print_px:
    print("Polynomial 1 coefficients (p1):")
    for i, val in enumerate(p1_coeffs):
        print(f"p1[{i}]: {val}")

    print("Polynomial 2 coefficients (p2):")
    for i, val in enumerate(p2_coeffs):
        print(f"p2[{i}]: {val}")

# Save coefficients to a JSON file in the 'data' subfolder
if save_coeffs_to_file:
    coeffs_dict = {
        "p1_coeffs": p1_coeffs.tolist(),
        "p2_coeffs": p2_coeffs.tolist()
    }
    with open("data/y_trajectory_coeffs_min_jerk.json", "w") as f:
        json.dump(coeffs_dict, f, indent=4)

# Export the coefficients for use in other scripts
def get_y_trajectory_coefficients():
    """Return the polynomial coefficients for y trajectory"""
    return p1_coeffs, p2_coeffs