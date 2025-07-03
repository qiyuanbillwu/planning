import numpy as np
import matplotlib.pyplot as plt
import json
from constants import T1, T2

# Helper to evaluate a polynomial at t (highest degree first)
def eval_poly(coeffs, t):
    # Reverse for np.polyval (expects highest degree first)
    return np.polyval(list(reversed(coeffs)), t)

# Load coefficients
with open('data/x_trajectory_coeffs_min_jerk.json', 'r') as f:
    x_coeffs = json.load(f)
with open('data/y_trajectory_coeffs_min_jerk.json', 'r') as f:
    y_coeffs = json.load(f)
with open('data/z_trajectory_coeffs_min_jerk.json', 'r') as f:
    z_coeffs = json.load(f)

# Each has p1_coeffs and p2_coeffs
x_p1, x_p2 = x_coeffs['p1_coeffs'], x_coeffs['p2_coeffs']
y_p1, y_p2 = y_coeffs['p1_coeffs'], y_coeffs['p2_coeffs']
z_p1, z_p2 = z_coeffs['p1_coeffs'], z_coeffs['p2_coeffs']

# Time arrays for each segment
N = 200
T_total = T1 + T2
t1 = np.linspace(0, T1, N)
t2 = np.linspace(0, T2, N)

# Evaluate polynomials for each segment
x1 = [eval_poly(x_p1, t) for t in t1]
x2 = [eval_poly(x_p2, t) for t in t2]
y1 = [eval_poly(y_p1, t) for t in t1]
y2 = [eval_poly(y_p2, t) for t in t2]
z1 = [eval_poly(z_p1, t) for t in t1]
z2 = [eval_poly(z_p2, t) for t in t2]

# Concatenate segments
t = np.concatenate([t1, t2 + T1])
x = np.concatenate([x1, x2])
y = np.concatenate([y1, y2])
z = np.concatenate([z1, z2])

# Plot
fig = plt.figure(figsize=(8,6), num=1)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=10, azim=160) # type: ignore
ax.plot(x, y, z, label='3D Trajectory', color='b')
ax.scatter([x[0], x[N-1], x[-1]], [y[0], y[N-1], y[-1]], [z[0], z[N-1], z[-1]], color='r', s=50, label='Key Points')  # type: ignore
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')  # type: ignore
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 4) # type: ignore
ax.set_title('3D Minimum Snap Trajectory')
ax.legend()
plt.show()
