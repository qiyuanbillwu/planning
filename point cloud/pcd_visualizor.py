import os
import sys
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

# Add parent directory to sys.path so utils can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_positions  # Now this should work
from waypoints import waypoint_list as waypoints  # Assuming waypoints are defined in a separate file

# Load the PCD file
pcd = o3d.io.read_point_cloud("/Users/wuqiyuan/Library/Mobile Documents/com~apple~CloudDocs/Desktop/UCLA/research/vectr/richter planning/point cloud/tilted_cylinders.pcd")
# o3d.visualization.draw_geometries([pcd])
print("Point cloud loaded successfully.")

# Convert the point cloud to a NumPy array
points = np.asarray(pcd.points)

# --- Load trajectory coefficients and time parameters ---
with open("/Users/wuqiyuan/Library/Mobile Documents/com~apple~CloudDocs/Desktop/UCLA/research/vectr/richter planning/data/polynomial_coefficients.json", "r") as f:
    coeffs = json.load(f)
with open("/Users/wuqiyuan/Library/Mobile Documents/com~apple~CloudDocs/Desktop/UCLA/research/vectr/richter planning/data/optimized_time_parameters.json", "r") as f:
    time_params = json.load(f)
Ts = np.array(time_params["optimized_times"])
segment_times = np.cumsum([0] + list(Ts))

waypoints = np.array(waypoints)

# --- Get trajectory points using utils.get_positions ---
t_traj, x_traj, y_traj, z_traj = get_positions(coeffs, Ts, segment_times, order=7)

# --- Plot 3D point cloud with trajectory and waypoints ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, alpha=0.5, label="Point Cloud")
ax.plot(x_traj, y_traj, z_traj, 'b-', linewidth=2, label="Trajectory")
ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='red', s=60, marker='o', label="Waypoints")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_xlim([-1.9, 1.9])
# ax.set_ylim([-1.9, 1.9])
# ax.set_zlim([0.1, 3.9])
ax.set_title("Trajectory and Waypoints on Point Cloud")
ax.legend()

# # --- Top-down view ---
# fig_top = plt.figure(figsize=(10, 10))
# ax_top = fig_top.add_subplot(111, projection='3d')
# ax_top.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.2)
# # ax_top.plot(x_traj, y_traj, z_traj, 'b-', linewidth=2, label="Trajectory")
# ax_top.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='red', s=60, marker='o', label="Waypoints")
# ax_top.set_xlabel('X')
# ax_top.set_ylabel('Y')
# ax_top.set_zlabel('Z')
# ax_top.set_title("Top-down view: Trajectory and Waypoints")
# ax_top.view_init(elev=90, azim=-90)

plt.show()

