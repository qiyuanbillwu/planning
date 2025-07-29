import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from utils import get_positions

r0 = np.array([1.5, -1, 2])
r1 = np.array([0.5, -1.2, 2])
r2 = np.array([-0.5, -0.95, 2])
r3 = np.array([-1, 0, 2])
r4 = np.array([-0.7, 1, 2])
r5 = np.array([0.5, 1.8, 2])
r6 = np.array([1.1, 0.75, 2])
r7 = np.array([1.5, -1, 2])
waypoints = [r0, r1, r2, r3, r4, r5, r6, r7]

# Load the polynomial coefficients
with open('data/polynomial_coefficients.json', 'r') as f:
    coeffs = json.load(f)

# Time parameters from eight_waypoints.py
# Load optimized time parameters
with open('data/optimized_time_parameters.json', 'r') as f:
    time_params = json.load(f)
Ts = np.array(time_params['optimized_times'])  # Duration of each segment from optimization
total_time = np.sum(Ts)

# Calculate segment times
segment_times = np.cumsum([0] + list(Ts))  # [0, 5, 10, 15, 20, 25, 30, 35]

# Use the get_positions function from utils
t_traj, x_traj, y_traj, z_traj = get_positions(coeffs, Ts, segment_times, order=5)

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
# configure orientation
ax.view_init(elev=17, azim=-118)  # type: ignore

# Plot the trajectory
ax.plot(x_traj, y_traj, z_traj, 'b-', linewidth=2, label='Trajectory')

# Plot waypoints
waypoints = np.array(waypoints)

ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
           c='red', s=100, marker='o', label='Waypoints') # type: ignore

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z') # type: ignore
ax.set_title('3D Trajectory with 7 Segments (Min-Jerk)')

# Add legend
ax.legend()

# Add grid
ax.grid(True)

# Print trajectory information
print(f"Trajectory Information:")
print(f"Total duration: {total_time} seconds")
print(f"Number of segments: 7")
print(f"Duration per segment: {Ts} seconds")
print(f"Start position: ({x_traj[0]:.3f}, {y_traj[0]:.3f}, {z_traj[0]:.3f})")
print(f"End position: ({x_traj[-1]:.3f}, {y_traj[-1]:.3f}, {z_traj[-1]:.3f})")
print(f"Total distance: {np.sum(np.sqrt(np.diff(x_traj)**2 + np.diff(y_traj)**2 + np.diff(z_traj)**2)):.3f} units")

# Show the plot
plt.tight_layout()
plt.show()

