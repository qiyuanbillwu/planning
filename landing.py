# 2 waypoints landing on a translating and oscillating platform

import numpy as np
from constants import v_avg
from utils import a, Q_snap
from scipy.linalg import solve_triangular
from plotting import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def compute_coeffs(tf, r0, v0, a0, j0, r_f, v_f, a_f, j_f):
    """
    Compute trajectory polynomial coefficients for given boundary conditions and time.
    """
    A = np.zeros((8, 8))
    for i in range(4):
        A[4 + i, :] = a(i, tf)
    A[0, 0] = 1
    A[1, 1] = 1
    A[2, 2] = 2
    A[3, 3] = 6

    b = np.zeros((8, 3))
    b[0, :] = r0
    b[1, :] = v0
    b[2, :] = a0
    b[3, :] = j0
    b[4, :] = r_f(tf)
    b[5, :] = v_f(tf)
    b[6, :] = a_f(tf)
    b[7, :] = j_f(tf)

    coeffs = np.linalg.solve(A, b)
    return coeffs
def r_f(t):
    return np.array([t, np.sin(t), 0])
def v_f(t):
    return np.array([1, np.cos(t), 0])
def a_f(t):
    return np.array([0, -np.sin(t), 0])
def j_f(t):
    return np.array([0, -np.cos(t), 0])

def calc_J(coeffs, tf, kT):
    # Compute the cost based on the coefficients and the time
    return coeffs[:,0].T @ Q_snap(tf) @ coeffs[:,0] + coeffs[:,1].T @ Q_snap(tf) @ coeffs[:,1] + coeffs[:,2].T @ Q_snap(tf) @ coeffs[:,2] + kT * tf

def gradient_descent_tf(tf_init, r0, v0, a0, j0, r_f, v_f, a_f, j_f, kT=10, alpha=1e-3, delta=1e-8, TOL=1e-3, ITER=2000):
    """
    Performs gradient descent to optimize tf for minimum cost.
    Returns the optimized tf.
    """
    tf = tf_init
    coeffs = compute_coeffs(tf, r0, v0, a0, j0, r_f, v_f, a_f, j_f)
    for i in range(ITER):
        J = calc_J(coeffs, tf, kT)
        t_delta = tf + delta
        coeffs_delta = compute_coeffs(t_delta, r0, v0, a0, j0, r_f, v_f, a_f, j_f)
        J_delta = calc_J(coeffs_delta, t_delta, kT)
        gradient = (J_delta - J) / delta

        dt = gradient * alpha

        if np.abs(dt) < TOL:
            print(f"Converged at iteration {i}, time: {tf}, cost: {J}")
            break

        tf -= dt
        tf = max(tf, 0.1)
        coeffs = compute_coeffs(tf, r0, v0, a0, j0, r_f, v_f, a_f, j_f)
        if i % 50 == 0:
            print(f"Iteration {i}, time: {tf}, cost: {J}")
    return tf

if __name__ == "__main__":

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

    # coeffs = compute_coeffs(tf, r0, v0, a0, j0, r_f, v_f, a_f, j_f)

    tf = gradient_descent_tf(tf, r0, v0, a0, j0, r_f, v_f, a_f, j_f)

    coeffs = compute_coeffs(tf, r0, v0, a0, j0, r_f, v_f, a_f, j_f)

    times = np.array([tf])
    rf = r_f(tf)
    waypoints = np.array([r0, rf])
    t_traj, x_traj, y_traj, z_traj = get_trajectory(coeffs, times, points_per_seg=1000)
    plot_traj(x_traj, y_traj, z_traj, waypoints)

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
    ax.scatter(*r_f(tf), color='green', s=60, label='End')
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

    # ani = FuncAnimation(fig, update, frames=len(x_traj), interval=10, blit=True)

    # ani.save('landing.mp4')

    # plot_position(coeffs, times)
    # plot_velocity(coeffs, times)
    # plot_acceleration(coeffs, times)
    # plot_jerk(coeffs, times)
    # plot_snap(coeffs, times)

    plt.show()


