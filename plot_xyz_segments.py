import numpy as np
import matplotlib.pyplot as plt
import json
from utils import get_a_dot_hat, cross_matrix, allocation_matrix, a5, evaluate_polynomial, evaluate_velocity, evaluate_acceleration, evaluate_jerk, evaluate_snap, compute_forces, plot_positions, plot_velocities, plot_accelerations, plot_jerks, plot_snaps, plot_forces, print_trajectory_info
from constants import J, l, d, m, g

# Load the polynomial coefficients
with open('data/polynomial_coefficients.json', 'r') as f:
    coeffs = json.load(f)

# Load optimized time parameters
with open('data/optimized_time_parameters.json', 'r') as f:
    optimized_params = json.load(f)

# Time parameters using optimized times
Ts = np.array(optimized_params['optimized_times'])  # Duration of each segment
total_time = optimized_params['total_time']

# Calculate segment times
segment_times = np.cumsum([0] + list(Ts))

print(f"Segment times: {segment_times}")

# Create all plots using the enhanced functions
fig1, ax1, (t_traj, x_traj, y_traj, z_traj) = plot_positions(coeffs, Ts, segment_times, order=7)
fig2, ax2, (t_vel, x_vel, y_vel, z_vel) = plot_velocities(coeffs, Ts, segment_times, order=7)
fig3, ax3, (t_acc, x_acc, y_acc, z_acc) = plot_accelerations(coeffs, Ts, segment_times, order=7)
fig4, ax4, (t_jerk, x_jerk, y_jerk, z_jerk) = plot_jerks(coeffs, Ts, segment_times, order=7)
fig5, ax5, (t_snap, x_snap, y_snap, z_snap) = plot_snaps(coeffs, Ts, segment_times, order=7)
fig6, ax6, (t_forces, f1, f2, f3, f4) = plot_forces(coeffs, Ts, segment_times, order=7)

# Print trajectory information
# print_trajectory_info(coeffs, Ts, total_time, order=5)

plt.show()