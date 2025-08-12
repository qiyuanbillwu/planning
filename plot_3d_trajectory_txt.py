import numpy as np
import matplotlib.pyplot as plt
from waypoints import waypoint_list as waypoints  # Assuming waypoints are defined in a separate file
from utils import get_positions, load_coeffs_txt, load_times_txt

def eval_poly(coeffs, t, order=7):
    # coeffs: shape (8, 3) for one segment, t: scalar or array
    powers = np.array([t**i for i in range(order+1)]).T  # shape: (len(t), 8)
    return powers @ coeffs  # shape: (len(t), 3)

def get_trajectory(coeffs, times, order=7, points_per_seg=100):
    n_seg = len(times)
    x_traj, y_traj, z_traj = [], [], []
    t_traj = []
    t_total = 0
    for i in range(n_seg):
        t_local = np.linspace(0, times[i], points_per_seg)
        segment_coeffs = coeffs[i*8:(i+1)*8, :]  # shape: (8, 3)
        xyz = eval_poly(segment_coeffs, t_local, order)
        x_traj.extend(xyz[:, 0])
        y_traj.extend(xyz[:, 1])
        z_traj.extend(xyz[:, 2])
        t_traj.extend(t_total + t_local)
        t_total += times[i]
    return np.array(t_traj), np.array(x_traj), np.array(y_traj), np.array(z_traj)

if __name__ == "__main__":
    coeffs = load_coeffs_txt('data/coeffs.txt')  # shape: (56, 3) for 7 segments, 8 coeffs each
    times = load_times_txt('data/times.txt')     # shape: (7,)

    t_traj, x_traj, y_traj, z_traj = get_trajectory(coeffs, times, order=7)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_traj, y_traj, z_traj, 'b-', linewidth=2, label='Trajectory')
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
           c='red', s=100, marker='o', label='Waypoints') # type: ignore
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory from coeffs.txt and times.txt')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()