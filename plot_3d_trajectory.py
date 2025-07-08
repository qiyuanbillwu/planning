import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

# Load the polynomial coefficients
with open('data/polynomial_coefficients.json', 'r') as f:
    coeffs = json.load(f)

# Time parameters from eight_waypoints.py
# Load optimized time parameters
with open('data/optimized_time_parameters.json', 'r') as f:
    time_params = json.load(f)
Ts = np.array(time_params['optimized_times'])  # Duration of each segment from optimization
total_time = np.sum(Ts)

# Function to evaluate 5th order polynomial
def evaluate_polynomial(t, coeffs):
    """
    Evaluate a 5th order polynomial: a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    """
    return (coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + 
            coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5)

# Generate time points for plotting
t_plot = np.linspace(0, total_time, 1000)
dt = total_time / 1000

# Initialize arrays to store trajectory points
x_traj = []
y_traj = []
z_traj = []
t_traj = []

# Plot each segment
segment_times = np.cumsum([0] + list(Ts))  # [0, 5, 10, 15, 20, 25, 30, 35]

for i in range(7):
    segment_start = segment_times[i]
    segment_end = segment_times[i + 1]
    
    # Get coefficients for this segment
    p_coeffs = coeffs[f'p{i+1}_coeffs']
    
    # Generate time points for this segment
    t_segment = np.linspace(0, Ts[i], 200)  # 200 points per segment
    
    # Evaluate polynomial for this segment
    x_segment = [evaluate_polynomial(t, p_coeffs['x']) for t in t_segment]
    y_segment = [evaluate_polynomial(t, p_coeffs['y']) for t in t_segment]
    z_segment = [evaluate_polynomial(t, p_coeffs['z']) for t in t_segment]
    
    # Add to trajectory arrays
    x_traj.extend(x_segment)
    y_traj.extend(y_segment)
    z_traj.extend(z_segment)
    t_traj.extend(t_segment + segment_start)

# Convert to numpy arrays
x_traj = np.array(x_traj)
y_traj = np.array(y_traj)
z_traj = np.array(z_traj)
t_traj = np.array(t_traj)

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(x_traj, y_traj, z_traj, 'b-', linewidth=2, label='Trajectory')

# Plot waypoints
waypoints = np.array([
    [18.2908, -12.9164, 0.5],      # r0
    [16.0048, -6.01777, 0.746351], # r1
    [9.74278, -4.28989, 3.58934],  # r2
    [2.32316, -1.06404, 1.57101],  # r3
    [-2.50561, 5.7747, 1.74195],   # r4
    [-5.96077, 10.9205, 1.32572],  # r5
    [-16.5275, 15.9659, 1.26184],  # r6
    [-19.8453, 12.2357, 0.5]       # r7
])

ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
           c='red', s=100, marker='o', label='Waypoints') # type: ignore

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z') # type: ignore
ax.set_title('3D Trajectory with 7 Segments (Min-Jerk)')

# Add legend
ax.legend()

# Add grid
ax.grid(True)

# Print trajectory information
print(f"Trajectory Information:")
print(f"Total duration: {total_time} seconds")
print(f"Number of segments: 7")
print(f"Duration per segment: {Ts} seconds")
print(f"Start position: ({x_traj[0]:.3f}, {y_traj[0]:.3f}, {z_traj[0]:.3f})")
print(f"End position: ({x_traj[-1]:.3f}, {y_traj[-1]:.3f}, {z_traj[-1]:.3f})")
print(f"Total distance: {np.sum(np.sqrt(np.diff(x_traj)**2 + np.diff(y_traj)**2 + np.diff(z_traj)**2)):.3f} units")

# Show the plot
plt.tight_layout()
plt.show()

