# landing problem with multiple waypoints
# This script computes a trajectory that lands on a translating and oscillating platform with multiple waypoints


import numpy as np
from constants import v_avg
from utils import a, Q_snap
from scipy.linalg import solve_triangular
from plotting import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from linear_algebra import *

def r_f(t):
    return np.array([t, np.sin(t), 0])
def v_f(t):
    return np.array([1, np.cos(t), 0])
def a_f(t):
    return np.array([0, -np.sin(t), 0])
def j_f(t):
    return np.array([0, -np.cos(t), 0])

def calc_cost(p, times, kT):
    # Calculate the cost J
    J = 0.0
    for i in range(len(times)):
        T = times[i]
        Q_i = Q_snap(T)
        p_i = p[i*8:(i+1)*8, :]
        J += (p_i[:, 0].T @ Q_i @ p_i[:, 0] + 
              p_i[:, 1].T @ Q_i @ p_i[:, 1] + 
              p_i[:, 2].T @ Q_i @ p_i[:, 2])
    J += kT * np.sum(times)  # Add time penalty
    return J

r0 = np.array([-100, 0, 20])
v0 = np.array([0, 0, 0])
a0 = np.array([0, 0, 0])
j0 = np.array([0, 0, 0])

v_avg = 10
tf = (200 + np.sqrt(40000 + 41600*(v_avg**2+1))) / (2 * (v_avg**2 + 1))
print(f"Initial guess for tf: {tf}")

# rf = r_f(tf)
# vf = v_f(tf)
# af = a_f(tf)
# jf = j_f(tf)

def generate_waypoints(r0, r_f, tf):
    T_segment = 4 # length of each segment in seconds
    n = round(tf / T_segment) # number of segments
    # print(f"Number of segments: {n}")
    waypoints = np.zeros((n+1, 3))
    waypoints[0, :] = r0
    for i in range(1, n):
        waypoints[i, :] = r0 + i * (r_f(tf) - r0) / n
    waypoints[n, :] = r_f(tf)
    return waypoints

# waypoints = generate_waypoints(r0, r_f, tf)

# print(f"Waypoints: {waypoints}")

# n = len(waypoints) - 1  # number of segments
# times = np.zeros(n)
# for i in range(n):
#     times[i] = tf / n

# b = create_b_with_bc(waypoints, v0, a0, j0, v_f(tf), a_f(tf), j_f(tf))
# M = create_M(times)
# print(M.shape, b.shape)
# c = solve_c(M, b)

def gradient_descent_tf(tf, r0, v0, a0, j0, r_f, v_f, a_f, j_f, kT=10, alpha=1e-3, delta=1e-8, TOL=1e-3, ITER=2000):
    """
    Performs gradient descent to optimize tf for minimum cost.
    Returns the optimized tf.
    """
    coeffs, times = compute_coeffs(tf, r0, v0, a0, j0, r_f, v_f, a_f, j_f)
    J = calc_cost(coeffs, times, kT)
    print(f"Initial cost: {J}, initial tf: {tf}")
    for i in range(ITER):
        J = calc_cost(coeffs, times, kT)
        t_delta = tf + delta
        coeffs_delta, times_delta = compute_coeffs(t_delta, r0, v0, a0, j0, r_f, v_f, a_f, j_f)
        J_delta = calc_cost(coeffs_delta, times_delta, kT)
        gradient = (J_delta - J) / delta

        dt = gradient * alpha

        if np.abs(dt) < TOL:
            print(f"Converged at iteration {i}, time: {tf}, cost: {J}")
            break

        tf -= dt
        tf = max(tf, 0.1)
        coeffs, times = compute_coeffs(tf, r0, v0, a0, j0, r_f, v_f, a_f, j_f)
        if i % 50 == 0:
            print(f"Iteration {i}, time: {tf}, cost: {J}")
    return tf

def compute_coeffs(tf, r0, v0, a0, j0, r_f, v_f, a_f, j_f):
    waypoints = generate_waypoints(r0, r_f, tf)
    b = create_b_with_bc(waypoints, v0, a0, j0, v_f(tf), a_f(tf), j_f(tf))
    n = len(waypoints) - 1  # number of segments
    times = create_times(tf, n)
    M = create_M(times)
    coeffs = solve_c(M, b)
    return coeffs, times

def create_times(tf, n):
    times = np.zeros(n)
    for i in range(n):
        times[i] = tf / n
    return times

tf = gradient_descent_tf(tf, r0, v0, a0, j0, r_f, v_f, a_f, j_f)
waypoints = generate_waypoints(r0, r_f, tf)
c, times = compute_coeffs(tf, r0, v0, a0, j0, r_f, v_f, a_f, j_f)
# print(c)

t_traj, x_traj, y_traj, z_traj = get_trajectory(c, times)

plot_traj(x_traj, y_traj, z_traj, waypoints)


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
ax.scatter(*r_f(tf), color='green', s=60, label='End')
ax.scatter(waypoints[1:-1, 0], waypoints[1:-1, 1], waypoints[1:-1, 2], 
           c='red', s=60, marker='o', label='Waypoints') # type: ignore
point, = ax.plot([], [], [], 'bo', markersize=8, label='Vehicle')
platform, = ax.plot([], [], [], 'k-', linewidth=2, label='Platform')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Landing Trajectory Animation')
ax.legend()
ax.grid(True)

ax.set_xlim([-60, 40])
ax.set_ylim([-5, 5])
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

ani.save('landing_many_waypoints.mp4')

# plot_position(c, times)
# plot_velocity(c, times)
# plot_acceleration(c, times)
# plot_jerk(c, times)
# plot_snap(c, times)

plt.show()


