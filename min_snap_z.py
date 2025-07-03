from utils import a, Q
from constants import T1, T2, g
import numpy as np
import matplotlib.pyplot as plt
import json

# Option to save coefficients to file
save_coeffs_to_file = True  # Set to False to skip saving


# Define the boundary conditions for Z direction
# initial position, velocity, acceleration, jerk
z0 = 0.5
v0 = 0
a0 = 0
j0 = 0  

# final position, velocity, acceleration, jerk
z2 = 0.5
v2 = 0
a2 = 0
j2 = 0

# gate states - FIXED values
z1 = 2
v1 = 0  # Fixed velocity at gate
a1 = -g  # Fixed acceleration at gate  
j1 = 0  # Fixed jerk at gate

# construct the total Hessian matrix
Q1 = Q(T1)
Q2 = Q(T2)
Q_total = np.block([
    [Q1, np.zeros_like(Q1)],
    [np.zeros_like(Q2), Q2]
])

# define the fixed/specified derivatives
# fix the position, velocity, acceleration, and jerk at the gate
df = np.array([[z0], [v0], [a0], [j0], [z2], [v2], [a2], [j2], [z1], [z1], [v1], [v1], [a1], [a1], [j1], [j1]])

# construct the A matrix
zeros_block = np.zeros((1, 8))
A = np.block([
    [a(0,0), zeros_block], # z0
    [a(1,0), zeros_block], # v0
    [a(2,0), zeros_block], # a0
    [a(3,0), zeros_block], # j0
    [zeros_block, a(0,T2)], # z2
    [zeros_block, a(1,T2)], # v2
    [zeros_block, a(2,T2)], # a2
    [zeros_block, a(3,T2)], # j2
    [a(0,T1), zeros_block], # z1 (end of seg 1)
    [zeros_block, a(0,0)], # z1 (start of seg 2)
    [a(1,T1), zeros_block], # v1 (end of seg 1)
    [zeros_block, a(1,0)], # v1 (start of seg 2)
    [a(2,T1), zeros_block], # a1 (end of seg 1)
    [zeros_block, a(2,0)], # a1 (start of seg 2)
    [a(3,T1), zeros_block], # j1 (end of seg 1)
    [zeros_block, a(3,0)], # j1 (start of seg 2)
])

# Compute the inverse of A
A_inv = np.linalg.inv(A)

R = A_inv.T @ Q_total @ A_inv

# Partition R based on the sizes of df and dp
n_f = df.shape[0]  # number of fixed variables
n_p = 8 * 2 - n_f  # number of free variables

R_FF = R[:n_f, :n_f]
R_FP = R[:n_f, n_f:n_f+n_p]
R_PF = R[n_f:n_f+n_p, :n_f]
R_PP = R[n_f:n_f+n_p, n_f:n_f+n_p]

dp_star = -np.linalg.inv(R_PP) @ R_FP.T @ df

print_dp_star = False  # Set this to False to suppress printing

if print_dp_star:
    print("Optimal free variables (dp_star) for Z trajectory:")
    labels = [
        "v1 (velocity at z1)", 
        "a1 (acceleration at z1)", 
        "j1 (jerk at z1)"
    ]
    for i, val in enumerate(dp_star.flatten()):
        if i < len(labels):
            print(f"{labels[i]}: {val}")
        else:
            print(f"dp_star[{i}]: {val}")

# Assemble the complete solution vector d from df and dp_star
d = np.vstack([df, dp_star])

pz = A_inv @ d

print_pz = False  # Set this to False to suppress printing

if print_pz:
    print("Solution vector pz (all polynomial coefficients for Z):")

# Extract coefficients for each polynomial
p1_coeffs = pz[:8].flatten()  # First 8 coefficients for polynomial 1
p2_coeffs = pz[8:].flatten()  # Last 8 coefficients for polynomial 2

if print_pz:
    print("Polynomial 1 coefficients (p1_z):")
    for i, val in enumerate(p1_coeffs):
        print(f"p1_z[{i}]: {val}")

    print("Polynomial 2 coefficients (p2_z):")
    for i, val in enumerate(p2_coeffs):
        print(f"p2_z[{i}]: {val}")

# Save coefficients to a JSON file in the 'data' subfolder
if save_coeffs_to_file:
    coeffs_dict = {
        "p1_coeffs": p1_coeffs.tolist(),
        "p2_coeffs": p2_coeffs.tolist()
    }
    with open("data/z_trajectory_coeffs.json", "w") as f:
        json.dump(coeffs_dict, f, indent=4)


# Export the coefficients for use in other scripts
def get_z_trajectory_coefficients():
    """Return the polynomial coefficients for z trajectory"""
    return p1_coeffs, p2_coeffs
