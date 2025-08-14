# landing on a translating and oscillating platform
# start with 2 waypoints

import numpy as np
from constants import v_avg
from utils import a
from scipy.linalg import solve_triangular
from plotting import get_trajectory, plot_traj, eval_poly
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

r0 = np.array([-100, 0, 20])
v0 = np.array([0, 0, 0])
a0 = np.array([0, 0, 0])
j0 = np.array([0, 0, 0])

def r_f(t):
    return np.array([t, np.sin(t), 0])
def v_f(t):
    return np.array([1, np.cos(t), 0])
def a_f(t):
    return np.array([0, -np.sin(t), 0])
def j_f(t):
    return np.array([0, -np.cos(t), 0])

# Calculate the time it takes to reach the target position
v_avg = 10
tf = (200 + np.sqrt(40000 + 41600*(v_avg**2+1))) / (2 * (v_avg**2 + 1))
print("Initial guess for t:", tf)

# calculate final waypoint
rf = r_f(tf)
vf = v_f(tf)
af = a_f(tf)
jf = j_f(tf)

print("Final waypoint:", rf)

waypoints = np.array([r0, rf])

A = np.zeros((8,8))
for i in range(4):
    A[4+i, :] = a(i, tf)
A[0,0] = 1
A[1,1] = 1
A[2,2] = 2
A[3,3] = 6

# print(np.array2string(A, formatter={'float_kind':lambda x: f"{x:.1e}"}))

b = np.zeros((8,3))
b[0,:] = r0
b[1,:] = v0
b[2,:] = a0
b[3,:] = j0
b[4,:] = rf
b[5,:] = vf
b[6,:] = af
b[7,:] = jf

coeffs = np.linalg.solve(A, b)

times = np.array([tf])
t_traj, x_traj, y_traj, z_traj = get_trajectory(coeffs, times, points_per_seg=1000)
# plot_traj(x_traj, y_traj, z_traj, waypoints)

# plt.figure(figsize=(10, 10))
# plt.plot(t_traj, x_traj)

def get_platform_vertices(center, size=2.0):
    """
    Returns the 4 corners of a square platform centered at 'center' (x, y, z),
    lying in the XY plane.
    """
    half = size / 2
    # corners in XY plane
    corners = np.array([
        [center[0] - half, center[1] - half, center[2]],
        [center[0] + half, center[1] - half, center[2]],
        [center[0] + half, center[1] + half, center[2]],
        [center[0] - half, center[1] + half, center[2]],
        [center[0] - half, center[1] - half, center[2]],  # close the loop
    ])
    return corners

# Animation setup
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_traj, y_traj, z_traj, 'gray', alpha=0.5, label='Trajectory')
ax.scatter(*r0, color='red', s=60, label='Start')
# ax.scatter(*rf, color='green', s=60, label='End')
point, = ax.plot([], [], [], 'bo', markersize=8, label='Vehicle')
platform, = ax.plot([], [], [], 'k-', linewidth=2, label='Platform')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Landing Trajectory Animation')
ax.legend()
ax.grid(True)

ax.set_xlim([-40, 15])
ax.set_ylim([-3, 3])
# ax.set_zlim([0, 10])

def update(frame):
    # Vehicle
    point.set_data([x_traj[frame]], [y_traj[frame]])
    point.set_3d_properties([z_traj[frame]])
    # Platform
    center = r_f(t_traj[frame])
    corners = get_platform_vertices(center)
    platform.set_data(corners[:, 0], corners[:, 1])
    platform.set_3d_properties(corners[:, 2])
    return point, platform

ani = FuncAnimation(fig, update, frames=len(x_traj), interval=10, blit=True)

ani.save('landing.mp4')

plt.show()


