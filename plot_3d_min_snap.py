import time
import numpy as np
import matplotlib.pyplot as plt
import json
from utils import plot_positions, plot_velocities, plot_accelerations, plot_jerks, plot_snaps, plot_forces

# Load optimized coefficients and time parameters
with open("data/min_snap_coeffs.json", "r") as f:
    coeffs = json.load(f)
with open("data/min_snap_time.json", "r") as f:
    time_params = json.load(f)

T1 = time_params["T1"]
T2 = time_params["T2"]

# Setup for plotting using utils functions
Ts = [T1, T2]
segment_times = np.cumsum([0] + Ts)
print(f"Segment times: {segment_times}")
points_per_segment = 500

# Create all plots using the enhanced functions
fig1, ax1, (t_traj, x_traj, y_traj, z_traj) = plot_positions(coeffs, Ts, segment_times, order=7)
fig2, ax2, (t_vel, x_vel, y_vel, z_vel) = plot_velocities(coeffs, Ts, segment_times, order=7)
fig3, ax3, (t_acc, x_acc, y_acc, z_acc) = plot_accelerations(coeffs, Ts, segment_times, order=7)
fig4, ax4, (t_jerk, x_jerk, y_jerk, z_jerk) = plot_jerks(coeffs, Ts, segment_times, order=7)
fig5, ax5, (t_snap, x_snap, y_snap, z_snap) = plot_snaps(coeffs, Ts, segment_times, order=7)
fig6, ax6, (t_forces, f1, f2, f3, f4) = plot_forces(coeffs, Ts, segment_times, order=7)

plt.show()


