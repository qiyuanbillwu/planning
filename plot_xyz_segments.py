import numpy as np
import matplotlib.pyplot as plt
import json
from utils import get_a_dot_hat, cross_matrix, allocation_matrix, a5
from constants import J, l, d, m, g

# Load the polynomial coefficients
with open('data/polynomial_coefficients.json', 'r') as f:
    coeffs = json.load(f)

# Load optimized time parameters
with open('data/optimized_time_parameters.json', 'r') as f:
    optimized_params = json.load(f)

# Time parameters using optimized times
Ts = np.array(optimized_params['optimized_times'])  # Duration of each segment
total_time = optimized_params['total_time']

# Function to evaluate 5th order polynomial
def evaluate_polynomial(t, coeffs):
    """
    Evaluate a 5th order polynomial: a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    """
    return (coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + 
            coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5)

# Function to evaluate velocity (1st derivative)
def evaluate_velocity(t, coeffs):
    """
    Evaluate velocity: a1 + 2*a2*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4
    """
    return (coeffs[1] + 2*coeffs[2]*t + 3*coeffs[3]*t**2 + 
            4*coeffs[4]*t**3 + 5*coeffs[5]*t**4)

# Function to evaluate acceleration (2nd derivative)
def evaluate_acceleration(t, coeffs):
    """
    Evaluate acceleration: 2*a2 + 6*a3*t + 12*a4*t^2 + 20*a5*t^3
    """
    return (2*coeffs[2] + 6*coeffs[3]*t + 12*coeffs[4]*t**2 + 20*coeffs[5]*t**3)

# Function to evaluate jerk (3rd derivative)
def evaluate_jerk(t, coeffs):
    """
    Evaluate jerk: 6*a3 + 24*a4*t + 60*a5*t^2
    """
    return (6*coeffs[3] + 24*coeffs[4]*t + 60*coeffs[5]*t**2)

# Function to evaluate snap (4th derivative)
def evaluate_snap(t, coeffs):
    """
    Evaluate snap: 24*a4 + 120*a5*t
    """
    return (24*coeffs[4] + 120*coeffs[5]*t)

# Function to compute forces using the same method as trajectory_calcs.py
def compute_forces(t, p_coeffs):
    """
    Compute motor forces at time t for a given segment using the same method as trajectory_calcs.py
    """
    # Get trajectory state at time t within the segment
    tau = t  # Time within the segment
    
    # Get position, velocity, acceleration, jerk using polynomial evaluation
    r = np.array([
        evaluate_polynomial(tau, p_coeffs['x']),
        evaluate_polynomial(tau, p_coeffs['y']),
        evaluate_polynomial(tau, p_coeffs['z'])
    ])
    
    v = np.array([
        evaluate_velocity(tau, p_coeffs['x']),
        evaluate_velocity(tau, p_coeffs['y']),
        evaluate_velocity(tau, p_coeffs['z'])
    ])
    
    a = np.array([
        evaluate_acceleration(tau, p_coeffs['x']),
        evaluate_acceleration(tau, p_coeffs['y']),
        evaluate_acceleration(tau, p_coeffs['z'])
    ])
    
    j = np.array([
        evaluate_jerk(tau, p_coeffs['x']),
        evaluate_jerk(tau, p_coeffs['y']),
        evaluate_jerk(tau, p_coeffs['z'])
    ])
    
    # Compute snap (4th derivative)
    s = np.array([
        evaluate_snap(tau, p_coeffs['x']),
        evaluate_snap(tau, p_coeffs['y']),
        evaluate_snap(tau, p_coeffs['z'])
    ])
    
    # Compute desired acceleration including gravity
    a_d = a + np.array([0, 0, g])
    a_d_hat = a_d / np.linalg.norm(a_d)
    T_d_hat = np.array([0, 0, 1])
    I = np.identity(3)
    
    # Compute rotation matrix and angular velocities
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
    
    # Compute torques and forces
    tau = J @ wdot + np.cross(w, J @ w)
    T_val = m * np.linalg.norm(a_d)
    tau_full = np.array([T_val, *tau])
    
    # Solve for motor forces using allocation matrix
    a_matrix = allocation_matrix(l, d)
    f = np.linalg.solve(a_matrix, tau_full)
    
    return f

# Initialize arrays to store trajectory points
x_traj = []
y_traj = []
z_traj = []
t_traj = []

# Initialize arrays to store velocity, acceleration, and jerk
x_vel = []
y_vel = []
z_vel = []
x_acc = []
y_acc = []
z_acc = []
x_jerk = []
y_jerk = []
z_jerk = []
x_snap = []
y_snap = []
z_snap = []

# Initialize arrays to store forces
f1 = []
f2 = []
f3 = []
f4 = []

# Plot each segment
segment_times = np.cumsum([0] + list(Ts))  

for i in range(7):
    segment_start = segment_times[i]
    segment_end = segment_times[i + 1]
    
    # Get coefficients for this segment
    p_coeffs = coeffs[f'p{i+1}_coeffs']
    
    # Generate time points for this segment
    t_segment = np.linspace(0, Ts[i]-0.1, 100)  # 200 points per segment
    
    # Evaluate polynomial for this segment
    x_segment = [evaluate_polynomial(t, p_coeffs['x']) for t in t_segment]
    y_segment = [evaluate_polynomial(t, p_coeffs['y']) for t in t_segment]
    z_segment = [evaluate_polynomial(t, p_coeffs['z']) for t in t_segment]
    
    # Evaluate velocity for this segment
    x_vel_segment = [evaluate_velocity(t, p_coeffs['x']) for t in t_segment]
    y_vel_segment = [evaluate_velocity(t, p_coeffs['y']) for t in t_segment]
    z_vel_segment = [evaluate_velocity(t, p_coeffs['z']) for t in t_segment]
    
    # Evaluate acceleration for this segment
    x_acc_segment = [evaluate_acceleration(t, p_coeffs['x']) for t in t_segment]
    y_acc_segment = [evaluate_acceleration(t, p_coeffs['y']) for t in t_segment]
    z_acc_segment = [evaluate_acceleration(t, p_coeffs['z']) for t in t_segment]
    
    # Evaluate jerk for this segment
    x_jerk_segment = [evaluate_jerk(t, p_coeffs['x']) for t in t_segment]
    y_jerk_segment = [evaluate_jerk(t, p_coeffs['y']) for t in t_segment]
    z_jerk_segment = [evaluate_jerk(t, p_coeffs['z']) for t in t_segment]
    
    # Compute snap for this segment
    x_snap_segment = [evaluate_snap(t, p_coeffs['x']) for t in t_segment]
    y_snap_segment = [evaluate_snap(t, p_coeffs['y']) for t in t_segment]
    z_snap_segment = [evaluate_snap(t, p_coeffs['z']) for t in t_segment]
    
    # Compute forces for this segment
    f1_segment = []
    f2_segment = []
    f3_segment = []
    f4_segment = []
    
    for t in t_segment:
        forces = compute_forces(t, p_coeffs)
        f1_segment.append(forces[0])
        f2_segment.append(forces[1])
        f3_segment.append(forces[2])
        f4_segment.append(forces[3])
    
    # Add to trajectory arrays
    x_traj.extend(x_segment)
    y_traj.extend(y_segment)
    z_traj.extend(z_segment)
    t_traj.extend(t_segment + segment_start)
    
    # Add to velocity, acceleration, and jerk arrays
    x_vel.extend(x_vel_segment)
    y_vel.extend(y_vel_segment)
    z_vel.extend(z_vel_segment)
    x_acc.extend(x_acc_segment)
    y_acc.extend(y_acc_segment)
    z_acc.extend(z_acc_segment)
    x_jerk.extend(x_jerk_segment)
    y_jerk.extend(y_jerk_segment)
    z_jerk.extend(z_jerk_segment)
    x_snap.extend(x_snap_segment)
    y_snap.extend(y_snap_segment)
    z_snap.extend(z_snap_segment)
    
    # Add to force arrays
    f1.extend(f1_segment)
    f2.extend(f2_segment)
    f3.extend(f3_segment)
    f4.extend(f4_segment)

# Convert to numpy arrays
x_traj = np.array(x_traj)
y_traj = np.array(y_traj)
z_traj = np.array(z_traj)
t_traj = np.array(t_traj)

# Convert velocity, acceleration, and jerk to numpy arrays
x_vel = np.array(x_vel)
y_vel = np.array(y_vel)
z_vel = np.array(z_vel)
x_acc = np.array(x_acc)
y_acc = np.array(y_acc)
z_acc = np.array(z_acc)
x_jerk = np.array(x_jerk)
y_jerk = np.array(y_jerk)
z_jerk = np.array(z_jerk)
x_snap = np.array(x_snap)
y_snap = np.array(y_snap)
z_snap = np.array(z_snap)

# Convert forces to numpy arrays
f1 = np.array(f1)
f2 = np.array(f2)
f3 = np.array(f3)
f4 = np.array(f4)

# Create separate figures for position, velocity, acceleration, jerk, snap, and forces
fig1, ax1 = plt.subplots(figsize=(14, 8))
fig2, ax2 = plt.subplots(figsize=(14, 8))
fig3, ax3 = plt.subplots(figsize=(14, 8))
fig4, ax4 = plt.subplots(figsize=(14, 8))
fig5, ax5 = plt.subplots(figsize=(14, 8)) # Snap figure
fig6, ax6 = plt.subplots(figsize=(14, 8)) # Forces figure

# Plot positions
ax1.plot(t_traj, x_traj, 'r-', linewidth=2, label='X Position')
ax1.plot(t_traj, y_traj, 'g-', linewidth=2, label='Y Position')
ax1.plot(t_traj, z_traj, 'b-', linewidth=2, label='Z Position')
ax1.set_ylabel('Position')
ax1.set_xlabel('Time (s)')
ax1.set_title('Position vs Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot velocities
ax2.plot(t_traj, x_vel, 'r-', linewidth=2, label='X Velocity')
ax2.plot(t_traj, y_vel, 'g-', linewidth=2, label='Y Velocity')
ax2.plot(t_traj, z_vel, 'b-', linewidth=2, label='Z Velocity')
ax2.set_ylabel('Velocity')
ax2.set_xlabel('Time (s)')
ax2.set_title('Velocity vs Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot accelerations
ax3.plot(t_traj, x_acc, 'r-', linewidth=2, label='X Acceleration')
ax3.plot(t_traj, y_acc, 'g-', linewidth=2, label='Y Acceleration')
ax3.plot(t_traj, z_acc, 'b-', linewidth=2, label='Z Acceleration')
ax3.set_ylabel('Acceleration')
ax3.set_xlabel('Time (s)')
ax3.set_title('Acceleration vs Time')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot jerks
ax4.plot(t_traj, x_jerk, 'r*', linewidth=2, label='X Jerk')
ax4.plot(t_traj, y_jerk, 'g-', linewidth=2, label='Y Jerk')
ax4.plot(t_traj, z_jerk, 'b-', linewidth=2, label='Z Jerk')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Jerk')
ax4.set_title('Jerk vs Time')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot snaps
ax5.plot(t_traj, x_snap, 'r*', linewidth=2, label='X Snap')
ax5.plot(t_traj, y_snap, 'g-', linewidth=2, label='Y Snap')
ax5.plot(t_traj, z_snap, 'b-', linewidth=2, label='Z Snap')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Snap')
ax5.set_title('Snap vs Time')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot forces
ax6.plot(t_traj, f1, 'r*', linewidth=2, label='Motor 1')
ax6.plot(t_traj, f2, 'g-', linewidth=2, label='Motor 2')
ax6.plot(t_traj, f3, 'b-', linewidth=2, label='Motor 3')
ax6.plot(t_traj, f4, 'orange', linewidth=2, label='Motor 4')
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Force (N)')
ax6.set_title('Motor Forces vs Time')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Add vertical lines at segment boundaries for all plots
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

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    for i, t in enumerate(segment_times):
        ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)

# Print trajectory information
print(f"Trajectory Information:")
print(f"Total duration: {total_time} seconds")
print(f"Number of segments: 7")
print(f"Duration per segment: {Ts} seconds")
print(f"Start position: ({x_traj[0]:.3f}, {y_traj[0]:.3f}, {z_traj[0]:.3f})")
print(f"End position: ({x_traj[-1]:.3f}, {y_traj[-1]:.3f}, {z_traj[-1]:.3f})")
print(f"X range: [{np.min(x_traj):.3f}, {np.max(x_traj):.3f}]")
print(f"Y range: [{np.min(y_traj):.3f}, {np.max(y_traj):.3f}]")
print(f"Z range: [{np.min(z_traj):.3f}, {np.max(z_traj):.3f}]")
print(f"\nVelocity Information:")
print(f"X velocity range: [{np.min(x_vel):.3f}, {np.max(x_vel):.3f}]")
print(f"Y velocity range: [{np.min(y_vel):.3f}, {np.max(y_vel):.3f}]")
print(f"Z velocity range: [{np.min(z_vel):.3f}, {np.max(z_vel):.3f}]")
print(f"\nAcceleration Information:")
print(f"X acceleration range: [{np.min(x_acc):.3f}, {np.max(x_acc):.3f}]")
print(f"Y acceleration range: [{np.min(y_acc):.3f}, {np.max(y_acc):.3f}]")
print(f"Z acceleration range: [{np.min(z_acc):.3f}, {np.max(z_acc):.3f}]")
print(f"\nJerk Information:")
print(f"X jerk range: [{np.min(x_jerk):.3f}, {np.max(x_jerk):.3f}]")
print(f"Y jerk range: [{np.min(y_jerk):.3f}, {np.max(y_jerk):.3f}]")
print(f"Z jerk range: [{np.min(z_jerk):.3f}, {np.max(z_jerk):.3f}]")
print(f"\nSnap Information:")
print(f"X snap range: [{np.min(x_snap):.3f}, {np.max(x_snap):.3f}]")
print(f"Y snap range: [{np.min(y_snap):.3f}, {np.max(y_snap):.3f}]")
print(f"Z snap range: [{np.min(z_snap):.3f}, {np.max(z_snap):.3f}]")
print(f"\nForce Information:")
print(f"Motor 1 force range: [{np.min(f1):.3f}, {np.max(f1):.3f}] N")
print(f"Motor 2 force range: [{np.min(f2):.3f}, {np.max(f2):.3f}] N")
print(f"Motor 3 force range: [{np.min(f3):.3f}, {np.max(f3):.3f}] N")
print(f"Motor 4 force range: [{np.min(f4):.3f}, {np.max(f4):.3f}] N") 

# Calculate jerk and snap at the 2nd waypoint (between segments 1 and 2)
print(f"\n" + "="*60)
print(f"JERK AND SNAP VALUES AT 2ND WAYPOINT (t = {segment_times[1]:.1f}s)")
print(f"="*60)

# Get coefficients for segments 1 and 2
p1_coeffs = coeffs['p1_coeffs']
p2_coeffs = coeffs['p2_coeffs']

# Time at the end of segment 1 (start of segment 2)
t_waypoint = Ts[0]  # This is the time at the 2nd waypoint

print(f"Time at 2nd waypoint: {t_waypoint:.1f} seconds")
print(f"\nJERK VALUES:")
print(f"Segment 1 (end) - X jerk: {evaluate_jerk(t_waypoint, p1_coeffs['x']):.10f}")
print(f"Segment 2 (start) - X jerk: {evaluate_jerk(0, p2_coeffs['x']):.10f}")
print(f"Difference: {abs(evaluate_jerk(t_waypoint, p1_coeffs['x']) - evaluate_jerk(0, p2_coeffs['x'])):.10f}")

print(f"Segment 1 (end) - Y jerk: {evaluate_jerk(t_waypoint, p1_coeffs['y']):.10f}")
print(f"Segment 2 (start) - Y jerk: {evaluate_jerk(0, p2_coeffs['y']):.10f}")
print(f"Difference: {abs(evaluate_jerk(t_waypoint, p1_coeffs['y']) - evaluate_jerk(0, p2_coeffs['y'])):.10f}")

print(f"Segment 1 (end) - Z jerk: {evaluate_jerk(t_waypoint, p1_coeffs['z']):.10f}")
print(f"Segment 2 (start) - Z jerk: {evaluate_jerk(0, p2_coeffs['z']):.10f}")
print(f"Difference: {abs(evaluate_jerk(t_waypoint, p1_coeffs['z']) - evaluate_jerk(0, p2_coeffs['z'])):.10f}")

print(f"\nSNAP VALUES:")
print(f"Segment 1 (end) - X snap: {evaluate_snap(t_waypoint, p1_coeffs['x']):.10f}")
print(f"Segment 2 (start) - X snap: {evaluate_snap(0, p2_coeffs['x']):.10f}")
print(f"Difference: {abs(evaluate_snap(t_waypoint, p1_coeffs['x']) - evaluate_snap(0, p2_coeffs['x'])):.10f}")

print(f"Segment 1 (end) - Y snap: {evaluate_snap(t_waypoint, p1_coeffs['y']):.10f}")
print(f"Segment 2 (start) - Y snap: {evaluate_snap(0, p2_coeffs['y']):.10f}")
print(f"Difference: {abs(evaluate_snap(t_waypoint, p1_coeffs['y']) - evaluate_snap(0, p2_coeffs['y'])):.10f}")

print(f"Segment 1 (end) - Z snap: {evaluate_snap(t_waypoint, p1_coeffs['z']):.10f}")
print(f"Segment 2 (start) - Z snap: {evaluate_snap(0, p2_coeffs['z']):.10f}")
print(f"Difference: {abs(evaluate_snap(t_waypoint, p1_coeffs['z']) - evaluate_snap(0, p2_coeffs['z'])):.10f}")

print(f"\n" + "="*60)


plt.show()