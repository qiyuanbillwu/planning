import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
from utils import get_positions, compute_forces

# Load optimized coefficients and time parameters
with open("data/min_snap_coeffs.json", "r") as f:
    coeffs = json.load(f)
with open("data/min_snap_time.json", "r") as f:
    time_params = json.load(f)

T1 = time_params["T1"]
T2 = time_params["T2"]

Ts = [T1, T2]
segment_times = np.cumsum([0] + Ts)
points_per_segment = 500

# Get trajectory data
t_traj, x_traj, y_traj, z_traj = get_positions(coeffs, Ts, segment_times, points_per_segment, order=7)

# Precompute thrust vectors at each time step
thrust_vectors = []
for idx, t in enumerate(t_traj):
    seg_idx = min(len(Ts)-1, np.searchsorted(segment_times, t, side='right') - 1)
    seg_time = t - segment_times[seg_idx]
    p_coeffs = coeffs[f'p{seg_idx+1}_coeffs']
    forces, adhat = compute_forces(seg_time, p_coeffs, order=7)
    thrust_vectors.append([x_traj[idx], y_traj[idx], z_traj[idx], adhat[0], adhat[1], adhat[2]])

thrust_vectors = np.array(thrust_vectors)

# Create 3D animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 4)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Minimum Snap Trajectory with Thrust Direction Animation')

line, = ax.plot([], [], [], 'b-', linewidth=2)
point, = ax.plot([], [], [], 'ro', markersize=8)
quiver = None

def init():
    global quiver
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    if quiver:
        quiver.remove()
    quiver = ax.quiver([], [], [], [], [], [], length=0.5, color='g')
    return line, point, quiver

def update(frame):
    global quiver
    line.set_data(x_traj[:frame], y_traj[:frame])
    line.set_3d_properties(z_traj[:frame])
    point.set_data([x_traj[frame]], [y_traj[frame]])
    point.set_3d_properties([z_traj[frame]])
    if quiver:
        quiver.remove()
    # Draw thrust direction at current position using adhat
    x, y, z, dx, dy, dz = thrust_vectors[frame]
    quiver = ax.quiver(x, y, z, dx, dy, dz, length=0.5, color='g', normalize=True)
    return line, point, quiver

ani = animation.FuncAnimation(fig, update, frames=len(t_traj), init_func=init,
                              interval=10, blit=False)

plt.show()