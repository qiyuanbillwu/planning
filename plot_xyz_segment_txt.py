import numpy as np
import matplotlib.pyplot as plt
from utils import load_coeffs_txt, load_times_txt, a

def eval_poly(coeffs, t, s=4):
    powers = np.vstack([t**i for i in range(2*s)]).T
    return powers @ coeffs

def eval_vel(coeffs, t, s=4):
    powers = np.vstack([i * t**(i-1) for i in range(1, 2*s)]).T
    # trim coeffs to match powers
    coeffs = coeffs[-powers.shape[1]:, :]
    return powers @ coeffs

def eval_acc(coeffs, t, s=4):
    powers = np.vstack([i * (i-1) * t**(i-2) for i in range(2, 2*s)]).T
    # trim coeffs to match powers
    coeffs = coeffs[-powers.shape[1]:, :]
    return powers @ coeffs

def eval_jerk(coeffs, t, s=4):
    powers = np.vstack([i * (i-1) * (i-2) * t**(i-3) for i in range(3, 2*s)]).T
    # trim coeffs to match powers
    coeffs = coeffs[-powers.shape[1]:, :]
    return powers @ coeffs

def eval_snap(coeffs, t, s=4): 
    powers = np.vstack([i * (i-1) * (i-2) * (i-3) * t**(i-4) for i in range(4, 2*s)]).T
    # trim coeffs to match powers
    coeffs = coeffs[-powers.shape[1]:, :]
    return powers @ coeffs

def plot_position(coeffs, times, s=4, points_per_seg=100):
    n_seg = len(times)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    t_total = 0
    for i in range(n_seg):
        t_local = np.linspace(0, times[i], points_per_seg)
        segment_coeffs = coeffs[i*8:(i+1)*8, :]
        xyz = eval_poly(segment_coeffs, t_local)
        ax.plot(t_total + t_local, xyz[:, 0], 'r-', label='X' if i==0 else None)
        ax.plot(t_total + t_local, xyz[:, 1], 'g-', label='Y' if i==0 else None)
        ax.plot(t_total + t_local, xyz[:, 2], 'b-', label='Z' if i==0 else None)
        t_total += times[i]
    # plot vertical lines for segments
    for i in range(n_seg+1):
        ax.axvline(x=np.sum(times[:i]), color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position')
    ax.set_title('Trajectory Position vs Time')
    ax.legend()
    plt.tight_layout()

def plot_velocity(coeffs, times, s=4, points_per_seg=100):
    n_seg = len(times)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    t_total = 0
    for i in range(n_seg):
        t_local = np.linspace(0, times[i], points_per_seg)
        segment_coeffs = coeffs[i*8:(i+1)*8, :]
        vel = eval_vel(segment_coeffs, t_local)
        ax.plot(t_total + t_local, vel[:, 0], 'r-', label='X' if i==0 else None)
        ax.plot(t_total + t_local, vel[:, 1], 'g-', label='Y' if i==0 else None)
        ax.plot(t_total + t_local, vel[:, 2], 'b-', label='Z' if i==0 else None)
        t_total += times[i]
    # plot vertical lines for segments
    for i in range(n_seg+1):
        ax.axvline(x=np.sum(times[:i]), color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity')
    ax.set_title('Trajectory Velocity vs Time')
    ax.legend()
    plt.tight_layout()

def plot_acceleration(coeffs, times, s=4, points_per_seg=1000):
    n_seg = len(times)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    t_total = 0
    for i in range(n_seg):
        t_local = np.linspace(0, times[i], points_per_seg)
        segment_coeffs = coeffs[i*8:(i+1)*8, :]
        acc = eval_acc(segment_coeffs, t_local)
        ax.plot(t_total + t_local, acc[:, 0], 'r-', label='X' if i==0 else None)
        ax.plot(t_total + t_local, acc[:, 1], 'g-', label='Y' if i==0 else None)
        ax.plot(t_total + t_local, acc[:, 2], 'b-', label='Z' if i==0 else None)
        t_total += times[i]
    # plot vertical lines for segments
    for i in range(n_seg+1):
        ax.axvline(x=np.sum(times[:i]), color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration')
    ax.set_title('Trajectory Acceleration vs Time')
    ax.legend()
    plt.tight_layout()

def plot_jerk(coeffs, times, s=4, points_per_seg=1000):
    n_seg = len(times)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    t_total = 0
    for i in range(n_seg):
        t_local = np.linspace(0, times[i], points_per_seg)
        segment_coeffs = coeffs[i*8:(i+1)*8, :]
        jerk = eval_jerk(segment_coeffs, t_local)
        ax.plot(t_total + t_local, jerk[:, 0], 'r-', label='X' if i==0 else None)
        ax.plot(t_total + t_local, jerk[:, 1], 'g-', label='Y' if i==0 else None)
        ax.plot(t_total + t_local, jerk[:, 2], 'b-', label='Z' if i==0 else None)
        t_total += times[i]
    # plot vertical lines for segments
    for i in range(n_seg+1):
        ax.axvline(x=np.sum(times[:i]), color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Jerk')
    ax.set_title('Trajectory Jerk vs Time')
    ax.legend()
    plt.tight_layout()

def plot_snap(coeffs, times, s=4, points_per_seg=1000):
    n_seg = len(times)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    t_total = 0
    for i in range(n_seg):
        t_local = np.linspace(0, times[i], points_per_seg)
        segment_coeffs = coeffs[i*8:(i+1)*8, :]
        snap = eval_snap(segment_coeffs, t_local)
        ax.plot(t_total + t_local, snap[:, 0], 'r-', label='X' if i==0 else None)
        ax.plot(t_total + t_local, snap[:, 1], 'g-', label='Y' if i==0 else None)
        ax.plot(t_total + t_local, snap[:, 2], 'b-', label='Z' if i==0 else None)
        t_total += times[i]
    # plot vertical lines for segments
    for i in range(n_seg+1):
        ax.axvline(x=np.sum(times[:i]), color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Snap')
    ax.set_title('Trajectory Snap vs Time')
    ax.legend()
    plt.tight_layout()


if __name__ == "__main__":
    coeffs = load_coeffs_txt('data/coeffs.txt')
    times = load_times_txt('data/times.txt')
    plot_position(coeffs, times)
    plot_velocity(coeffs, times)
    plot_acceleration(coeffs, times)
    plot_jerk(coeffs, times)
    plot_snap(coeffs, times)
    plt.show()