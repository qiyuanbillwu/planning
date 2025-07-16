# This file contains the calculations for the trajectory of the drone

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from utils import get_a_dot_hat, cross_matrix, allocation_matrix, a, a5
from constants import J, l, d, m, g, T1, T2

# === Option to select trajectory type ===
# Set TRAJ_TYPE to 'snap' or 'jerk'. You can also set via command line: python trajectory_calcs.py jerk
TRAJ_TYPE = 'jerk'
if len(sys.argv) > 1:
    if sys.argv[1].lower() in ['snap', 'jerk']:
        TRAJ_TYPE = sys.argv[1].lower()
    else:
        print("Unknown trajectory type, using default 'snap'.")

if TRAJ_TYPE == 'jerk':
    with open('data/x_trajectory_coeffs_min_jerk.json', 'r') as f:
        x_coeffs = json.load(f)
    with open('data/y_trajectory_coeffs_min_jerk.json', 'r') as f:
        y_coeffs = json.load(f)
    with open('data/z_trajectory_coeffs_min_jerk.json', 'r') as f:
        z_coeffs = json.load(f)
    poly_eval = a5
    coeff_len = 6
else:
    with open('data/x_trajectory_coeffs.json', 'r') as f:
        x_coeffs = json.load(f)
    with open('data/y_trajectory_coeffs.json', 'r') as f:
        y_coeffs = json.load(f)
    with open('data/z_trajectory_coeffs.json', 'r') as f:
        z_coeffs = json.load(f)
    poly_eval = a
    coeff_len = 8

x_p1, x_p2 = x_coeffs['p1_coeffs'], x_coeffs['p2_coeffs']
y_p1, y_p2 = y_coeffs['p1_coeffs'], y_coeffs['p2_coeffs']
z_p1, z_p2 = z_coeffs['p1_coeffs'], z_coeffs['p2_coeffs']


def get_traj_coeffs(t):
    if t < T1:
        return x_p1, y_p1, z_p1
    else:
        return x_p2, y_p2, z_p2

def get_traj_state(t):
    x_coeff, y_coeff, z_coeff = get_traj_coeffs(t)
    tau = t if t < T1 else t-T1
    r = np.array([
        poly_eval(0, tau) @ np.array(x_coeff),
        poly_eval(0, tau) @ np.array(y_coeff),
        poly_eval(0, tau) @ np.array(z_coeff)
    ]).flatten()
    v = np.array([
        poly_eval(1, tau) @ np.array(x_coeff),
        poly_eval(1, tau) @ np.array(y_coeff),
        poly_eval(1, tau) @ np.array(z_coeff)
    ]).flatten()
    a_vec = np.array([
        poly_eval(2, tau) @ np.array(x_coeff),
        poly_eval(2, tau) @ np.array(y_coeff),
        poly_eval(2, tau) @ np.array(z_coeff)
    ]).flatten()
    j = np.array([
        poly_eval(3, tau) @ np.array(x_coeff),
        poly_eval(3, tau) @ np.array(y_coeff),
        poly_eval(3, tau) @ np.array(z_coeff)
    ]).flatten()
    if TRAJ_TYPE == 'snap':
        s = np.array([
            poly_eval(4, tau) @ np.array(x_coeff),
            poly_eval(4, tau) @ np.array(y_coeff),
            poly_eval(4, tau) @ np.array(z_coeff)
        ]).flatten()
    else:
        s = np.zeros(3)
    return r, v, a_vec, j, s

a_matrix = allocation_matrix(l, d)

def compute_full_state(t):
    r, v, a, j, s = get_traj_state(t)
    a_d = a + np.array([0, 0, g])
    a_d_hat = a_d / np.linalg.norm(a_d)
    T_d_hat = np.array([0, 0, 1])
    I = np.identity(3)
    theta = np.arccos(np.clip(np.dot(T_d_hat, a_d_hat), -1.0, 1.0))
    if np.allclose(a_d_hat, T_d_hat):
        n_hat = np.array([0, 0, 1])
        w = np.zeros(3)
        wdot = np.zeros(3)
        q_d = np.concatenate(([np.cos(theta/2)], n_hat*np.sin(theta)))
    else:
        n = np.cross(T_d_hat, a_d_hat)
        n_hat = n / np.linalg.norm(n)
        n_cross = cross_matrix(n_hat)
        R_d = I + np.sin(theta) * n_cross + (1-np.cos(theta)) * n_cross @ n_cross
        q_d = np.concatenate(([np.cos(theta/2)], n_hat*np.sin(theta)))
        a_hat_dot = get_a_dot_hat(a_d, j)
        w = R_d.T @ a_hat_dot
        wx = -w[1]
        w[1] = w[0]
        w[0] = wx
        w[2] = 0
        a_hat_doubledot = s / np.linalg.norm(a_d) - (2 * j * (a_d @ j) + a_d * (j @ j + a_d @ s)) / np.linalg.norm(a_d)**3 + 3 * a_d * (a_d @ j)**2 / np.linalg.norm(a_d)**5
        wdot = R_d.T @ a_hat_doubledot - cross_matrix(w) @ R_d.T @ a_hat_dot
        wdotx = -wdot[1]
        wdot[1] = wdot[0]
        wdot[0] = wdotx
        wdot[2] = 0
    tau = J @ wdot + np.cross(w, J @ w)
    T_val = m * np.linalg.norm(a_d)
    tau_full = np.array([T_val, *tau])
    f = np.linalg.solve(a_matrix, tau_full)
    return {
        'r': r, 'v': v, 'a': a, 'j': j, 's': s,
        'q': q_d, 'w': w, 'wdot': wdot, 'f': f
    }

dt = 0.01
T_total = T1 + T2
positions = []
velocities = []
accelerations = []
jerks = []
snaps = []
forces = []

ts = np.arange(0, T_total, dt)
for t in ts:
    state = compute_full_state(t)
    positions.append(state['r'])
    velocities.append(state['v'])
    accelerations.append(state['a'])
    jerks.append(state['j'])
    snaps.append(state['s'])
    forces.append(state['f'])
positions = np.array(positions)
velocities = np.array(velocities)
accelerations = np.array(accelerations)
jerks = np.array(jerks)
snaps = np.array(snaps)
forces = np.array(forces)

if __name__ == "__main__":
    # Plot x, y, z position in one plot
    plt.figure(figsize=(10,6), num=1)
    plt.plot(ts, positions[:,0], label='x')
    plt.plot(ts, positions[:,1], label='y')
    plt.plot(ts, positions[:,2], label='z')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Position vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot velocity components
    plt.figure(figsize=(10,6), num=2)
    plt.plot(ts, velocities[:,0], label='v_x')
    plt.plot(ts, velocities[:,1], label='v_y')
    plt.plot(ts, velocities[:,2], label='v_z')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot acceleration components
    plt.figure(figsize=(10,6), num=3)
    plt.plot(ts, accelerations[:,0], label='a_x')
    plt.plot(ts, accelerations[:,1], label='a_y')
    plt.plot(ts, accelerations[:,2], label='a_z')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.title('Acceleration vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot jerk components
    plt.figure(figsize=(10,6), num=4)
    plt.plot(ts, jerks[:,0], label='j_x')
    plt.plot(ts, jerks[:,1], label='j_y')
    plt.plot(ts, jerks[:,2], label='j_z')
    plt.xlabel('Time (s)')
    plt.ylabel('Jerk (m/s^3)')
    plt.title('Jerk vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot snap components
    plt.figure(figsize=(10,6), num=5)
    plt.plot(ts, snaps[:,0], label='s_x')
    plt.plot(ts, snaps[:,1], label='s_y')
    plt.plot(ts, snaps[:,2], label='s_z')
    plt.xlabel('Time (s)')
    plt.ylabel('Snap (m/s^4)')
    plt.title('Snap vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot motor forces (existing code)
    plt.figure(figsize=(10,6))
    for i in range(4):
        plt.plot(ts, forces[:,i], label=f'Motor {i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Individual Motor Forces Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

