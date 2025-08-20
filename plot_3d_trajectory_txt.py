import numpy as np
import matplotlib.pyplot as plt
from waypoints import waypoint_list as waypoints  # Assuming waypoints are defined in a separate file
from utils import load_coeffs_txt, load_times_txt
from plotting import get_trajectory, plot_traj



if __name__ == "__main__":
    coeffs = load_coeffs_txt('data/coeffs.txt')  # shape: (56, 3) for 7 segments, 8 coeffs each
    times = load_times_txt('data/times.txt')     # shape: (7,)

    t_traj, x_traj, y_traj, z_traj = get_trajectory(coeffs, times)

    plot_traj(x_traj, y_traj, z_traj, waypoints)
    plt.show()