from utils import load_coeffs_txt, load_times_txt, a
from plotting import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    coeffs = load_coeffs_txt('data/coeffs.txt')
    times = load_times_txt('data/times.txt')
    plot_position(coeffs, times)
    plot_velocity(coeffs, times)
    plot_acceleration(coeffs, times)
    plot_jerk(coeffs, times)
    plot_snap(coeffs, times)
    plt.show()