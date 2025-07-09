import numpy as np
import matplotlib.pyplot as plt
from eight_waypoints import optimize_times_gradient_descent, get_coefficients, save_optimization_results, plot_optimization_history
from utils import evaluate_velocity, plot_velocities, get_velocities
import json

# Extract 8 waypoints from waypoints.txt
r0 = np.array([18.2908, -12.9164, 0.5])
r1 = np.array([16.0048, -6.01777, 0.746351])
r2 = np.array([9.74278, -4.28989, 3.58934])
r3 = np.array([2.32316, -1.06404, 1.57101])
r4 = np.array([-2.50561, 5.7747, 1.74195])
r5 = np.array([-5.96077, 10.9205, 1.32572])
r6 = np.array([-16.5275, 15.9659, 1.26184])
r7 = np.array([-19.8453, 12.2357, 0.5])

# set average and maximum velocity/acceleration
v_max = 5.0
v_avg = 3.0

# calculate distance between waypoints
d1 = np.linalg.norm(r1 - r0)
d2 = np.linalg.norm(r2 - r1)
d3 = np.linalg.norm(r3 - r2)
d4 = np.linalg.norm(r4 - r3)
d5 = np.linalg.norm(r5 - r4)
d6 = np.linalg.norm(r6 - r5)
d7 = np.linalg.norm(r7 - r6)

# calculate time for each segment basead on average velocity
t1 = d1 / v_avg
t2 = d2 / v_avg
t3 = d3 / v_avg
t4 = d4 / v_avg
t5 = d5 / v_avg
t6 = d6 / v_avg
t7 = d7 / v_avg

Ts_initial = np.array([t1, t2, t3, t4, t5, t6, t7])

# set gradient descent parameters
kT = 100
alpha = 10**(-4)
ITER = 1000
TOL = 10**(-3)

# run gradient descent
results = optimize_times_gradient_descent(Ts_initial, kT, alpha, max_iterations=ITER, tolerance=TOL)

# Save optimized coefficients and time parameters
print("\nSaving optimized coefficients and time parameters...")
coefficients = get_coefficients(results['Ts_optimized'])
save_optimization_results(results, coefficients)

# Plot optimization history
plot_optimization_history(results)

# evaluate the trajectory

# Load the polynomial coefficients
with open('data/polynomial_coefficients.json', 'r') as f:
    coeffs = json.load(f)

# Load optimized time parameters
with open('data/optimized_time_parameters.json', 'r') as f:
    optimized_params = json.load(f)

# Time parameters using optimized times
Ts = np.array(optimized_params['optimized_times'])  # Duration of each segment
total_time = optimized_params['total_time']
segment_times = np.cumsum([0] + list(Ts))  

fig2, ax2, (t_vel, x_vel, y_vel, z_vel) = plot_velocities(coeffs, Ts, segment_times)

# get the maximum absolute velocity in x, y, z
max_vel = np.max(np.abs(np.array([x_vel, y_vel, z_vel])))
# print(f"Maximum absolute velocity: {max_vel} m/s")

# if max_vel > v_max, scale the time parameters
scale_factor = 1.1
for _ in range(20):
    Ts = Ts * scale_factor
    coeffs = get_coefficients(Ts)
    segment_times = np.cumsum([0] + list(Ts))  
    # Load the updated coefficients from the JSON file
    with open('data/polynomial_coefficients.json', 'r') as f:
        coeffs = json.load(f)
    _, x_vel, y_vel, z_vel = get_velocities(coeffs, Ts, segment_times)
    max_vel = np.max(np.abs(np.array([x_vel, y_vel, z_vel])))
    if max_vel <= v_max:
        break

# update the optimized time parameters
optimized_params['optimized_times'] = Ts.tolist()
optimized_params['total_time'] = np.sum(Ts)
# save the updated optimized time parameters
with open('data/optimized_time_parameters.json', 'w') as f:
    json.dump(optimized_params, f)

# plot the new velocity
fig3, ax3, (t_vel, x_vel, y_vel, z_vel) = plot_velocities(coeffs, Ts, segment_times)

plt.show()