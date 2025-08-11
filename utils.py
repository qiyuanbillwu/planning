import numpy as np
import matplotlib.pyplot as plt
from constants import J, l, d, m, g

def a(n, t):
    """
    Returns the row vector for the nth derivative of a 7th order polynomial at time t.
    Used for minimum snap (7th order) trajectory generation.
    Coefficients: [t^0, t^1, ..., t^7]
    n: derivative order (0=pos, 1=vel, 2=acc, 3=jerk, 4=snap)
    """
    if t < 0:
        raise ValueError("t must be non-negative")
    if n == 0: # position
        return np.array([[1, t, t**2, t**3, t**4, t**5, t**6, t**7]])
    elif n == 1: # velocity
        return np.array([[0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4, 6*t**5, 7*t**6]])
    elif n == 2: # acceleration
        return np.array([[0, 0, 2, 6*t, 12*t**2, 20*t**3, 30*t**4, 42*t**5]])
    elif n == 3: # jerk
        return np.array([[0, 0, 0, 6, 24*t, 60*t**2, 120*t**3, 210*t**4]])
    elif n == 4: # snap
        return np.array([[0, 0, 0, 0, 24, 120*t, 360*t**2, 840*t**3]])
    elif n == 5: # crackle
        return np.array([[0, 0, 0, 0, 0, 120, 720*t, 2520*t**2]])
    elif n == 6: # pop
        return np.array([[0, 0, 0, 0, 0, 0, 720, 5040*t]])
    else:
        raise ValueError("n must be between 0 and 6 inclusive")

# New utility function for 5th order (min-jerk) polynomial
# This is for minimum jerk trajectory generation, with 6 coefficients: [t^0, t^1, ..., t^5]
def a5(n, t):
    """
    Returns the row vector for the nth derivative of a 5th order polynomial at time t.
    Used for minimum jerk (5th order) trajectory generation.
    Coefficients: [t^0, t^1, ..., t^5]
    n: derivative order (0=pos, 1=vel, 2=acc, 3=jerk)
    """
    if t < 0:
        raise ValueError("t must be non-negative")
    if n == 0: # position
        return np.array([[1, t, t**2, t**3, t**4, t**5]])
    elif n == 1: # velocity
        return np.array([[0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4]])
    elif n == 2: # acceleration
        return np.array([[0, 0, 2, 6*t, 12*t**2, 20*t**3]])
    elif n == 3: # jerk
        return np.array([[0, 0, 0, 6, 24*t, 60*t**2]])
    else:
        raise ValueError("n must be between 0 and 3 inclusive for 5th order polynomial")

def beta(i):
    if not (0 <= i <= 7):
        raise ValueError("i must be between 0 and 7 inclusive")
    if i <= 3:
        return 0
    return i*(i-1)*(i-2)*(i-3)

def Q_snap(T):
    """
    Returns the cost matrix for minimum snap (7th order) trajectory generation.
    Integrates the square of the 4th derivative (snap) over [0, T].
    Output: 8x8 matrix for 7th order polynomial.
    """
    Q_mat = np.zeros((8, 8))
    for i in range(4, 8):
        for j in range(4, 8):
            Q_mat[i, j] = beta(i) * beta(j) * T**(i + j - 7) / (i + j - 7)
    return Q_mat

def D_snap(T):

    D_mat = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            D_mat[i, j] = beta(i + 4) * beta(j + 4) * T**(i + j - 4) / (i + j - 4)
    return D_mat

# New cost matrix for 5th order (min-jerk) polynomial
# This is for minimum jerk trajectory generation, integrating the square of the 3rd derivative (jerk)
def Q_jerk(T):
    """
    Returns the cost matrix for minimum jerk (5th order) trajectory generation.
    Integrates the square of the 3rd derivative (jerk) over [0, T].
    Output: 6x6 matrix for 5th order polynomial.
    """
    Q_mat = np.zeros((6, 6))
    # For 5th order, jerk is the 3rd derivative, so start from i, j = 3
    def beta5(i):
        if not (0 <= i <= 5):
            raise ValueError("i must be between 0 and 5 inclusive")
        if i <= 2:
            return 0
        return i*(i-1)*(i-2)
    for i in range(3, 6):
        for j in range(3, 6):
            Q_mat[i, j] = beta5(i) * beta5(j) * T**(i + j - 5) / (i + j - 5)
    return Q_mat

# function to calculate the rate of change of a unit vector a_hat
def get_a_dot_hat(a, adot):
    if np.linalg.norm(a) == 0:
        raise ValueError("Input vector 'a' must not be the zero vector")

    return adot / np.linalg.norm(a) - a * (a.T @ adot) / np.linalg.norm(a)**3

def cross_matrix(v):
    """
    Returns the skew-symmetric cross-product matrix of a 3D vector v.
    
    Parameters:
        v : np.ndarray or list-like of shape (3,)
            A 3D vector [vx, vy, vz]
    
    Returns:
        np.ndarray : 3x3 skew-symmetric matrix such that cross(a, v) == cross_matrix(v) @ a
    """
    vx, vy, vz = v
    return np.array([
        [ 0,   -vz,  vy],
        [ vz,   0,  -vx],
        [-vy,  vx,   0 ]
    ])

def allocation_matrix(l,d):
    #  Front
    #    ^
    #    |
    # 1      2
    #    |
    # 4      3

    # 1 CCW
    # 2 CW
    # 3 CCW
    # 4 CW

    return np.array([
    [1, 1, 1, 1],        # Total thrust
    [l, l, -l, -l],      # Roll
    [l, -l, -l, l],      # Pitch
    [-d, d, -d, d]       # Yaw
    ])

# Function to evaluate 5th order polynomial
def evaluate_polynomial(t, coeffs, order=5):
    """
    Evaluate a polynomial of specified order at time t.
    order: 5 for 5th order (6 coefficients), 7 for 7th order (8 coefficients)
    """
    if order == 5:
        return sum([coeffs[i] * t**i for i in range(6)])
    elif order == 7:
        return sum([coeffs[i] * t**i for i in range(8)])
    else:
        raise ValueError("Order must be 5 or 7.")

# Function to evaluate velocity (1st derivative)
def evaluate_velocity(t, coeffs, order=5):
    """
    Evaluate velocity: a1 + 2*a2*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4
    """
    if order == 5:
        return (coeffs[1] + 2*coeffs[2]*t + 3*coeffs[3]*t**2 +
                4*coeffs[4]*t**3 + 5*coeffs[5]*t**4)
    elif order == 7:
        return (coeffs[1] + 2*coeffs[2]*t + 3*coeffs[3]*t**2 +
                4*coeffs[4]*t**3 + 5*coeffs[5]*t**4 +
                6*coeffs[6]*t**5 + 7*coeffs[7]*t**6)
    else:
        raise ValueError("Order must be 5 or 7.")

# Function to evaluate acceleration (2nd derivative)
def evaluate_acceleration(t, coeffs, order=5):
    """
    Evaluate acceleration: 2*a2 + 6*a3*t + 12*a4*t^2 + 20*a5*t^3
    """
    if order == 5:
        return (2*coeffs[2] + 6*coeffs[3]*t + 12*coeffs[4]*t**2 + 20*coeffs[5]*t**3)
    elif order == 7:
        return (2*coeffs[2] + 6*coeffs[3]*t + 12*coeffs[4]*t**2 + 20*coeffs[5]*t**3 +
                30*coeffs[6]*t**4 + 42*coeffs[7]*t**5)
    else:
        raise ValueError("Order must be 5 or 7.")

# Function to evaluate jerk (3rd derivative)
def evaluate_jerk(t, coeffs, order=5):
    """
    Evaluate jerk: 6*a3 + 24*a4*t + 60*a5*t^2
    """
    if order == 5:
        return (6*coeffs[3] + 24*coeffs[4]*t + 60*coeffs[5]*t**2)
    elif order == 7:
        return (6*coeffs[3] + 24*coeffs[4]*t + 60*coeffs[5]*t**2 +
                120*coeffs[6]*t**3 + 210*coeffs[7]*t**4)
    else:
        raise ValueError("Order must be 5 or 7.")

# Function to evaluate snap (4th derivative)
def evaluate_snap(t, coeffs, order=5):
    """
    Evaluate snap: 24*a4 + 120*a5*t
    """
    if order == 5:
        return (24*coeffs[4] + 120*coeffs[5]*t)
    elif order == 7:
        return (24*coeffs[4] + 120*coeffs[5]*t +
                360*coeffs[6]*t**2 + 840*coeffs[7]*t**3)
    else:
        raise ValueError("Order must be 5 or 7.")

# Function to compute forces using the same method as trajectory_calcs.py
def compute_forces(t, p_coeffs, order=7):
    """
    Compute motor forces at time t for a given segment using the same method as trajectory_calcs.py
    Supports both 5th and 7th order polynomials.
    """
    # Get trajectory state at time t within the segment
    tau = t  # Time within the segment

    # Get position, velocity, acceleration, jerk using polynomial evaluation
    r = np.array([
        evaluate_polynomial(tau, p_coeffs['x'], order=order),
        evaluate_polynomial(tau, p_coeffs['y'], order=order),
        evaluate_polynomial(tau, p_coeffs['z'], order=order)
    ])

    v = np.array([
        evaluate_velocity(tau, p_coeffs['x'], order=order),
        evaluate_velocity(tau, p_coeffs['y'], order=order),
        evaluate_velocity(tau, p_coeffs['z'], order=order)
    ])

    a = np.array([
        evaluate_acceleration(tau, p_coeffs['x'], order=order),
        evaluate_acceleration(tau, p_coeffs['y'], order=order),
        evaluate_acceleration(tau, p_coeffs['z'], order=order)
    ])

    j = np.array([
        evaluate_jerk(tau, p_coeffs['x'], order=order),
        evaluate_jerk(tau, p_coeffs['y'], order=order),
        evaluate_jerk(tau, p_coeffs['z'], order=order)
    ])

    # Compute snap (4th derivative)
    s = np.array([
        evaluate_snap(tau, p_coeffs['x'], order=order),
        evaluate_snap(tau, p_coeffs['y'], order=order),
        evaluate_snap(tau, p_coeffs['z'], order=order)
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

def plot_positions(coeffs, Ts, segment_times, points_per_segment=200, order=7):
    """Plot position vs time for all three axes"""
    t_traj, x_traj, y_traj, z_traj = get_positions(coeffs, Ts, segment_times, points_per_segment, order=order)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(t_traj, x_traj, 'r-', linewidth=2, label='X Position')
    ax.plot(t_traj, y_traj, 'g-', linewidth=2, label='Y Position')
    ax.plot(t_traj, z_traj, 'b-', linewidth=2, label='Z Position')
    ax.set_ylabel('Position')
    ax.set_xlabel('Time (s)')
    ax.set_title('Position vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    for t in segment_times:
        ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
    return fig, ax, (t_traj, x_traj, y_traj, z_traj)

def plot_velocities(coeffs, Ts, segment_times, points_per_segment=200, order=7):
    """Plot velocity vs time for all three axes"""
    t_traj, x_vel, y_vel, z_vel = get_velocities(coeffs, Ts, segment_times, points_per_segment, order=order)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(t_traj, x_vel, 'r-', linewidth=2, label='X Velocity')
    ax.plot(t_traj, y_vel, 'g-', linewidth=2, label='Y Velocity')
    ax.plot(t_traj, z_vel, 'b-', linewidth=2, label='Z Velocity')
    ax.set_ylabel('Velocity')
    ax.set_xlabel('Time (s)')
    ax.set_title('Velocity vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    for t in segment_times:
        ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
    return fig, ax, (t_traj, x_vel, y_vel, z_vel)

def plot_accelerations(coeffs, Ts, segment_times, points_per_segment=200, order=7):
    """Plot acceleration vs time for all three axes"""
    t_traj, x_acc, y_acc, z_acc = get_accelerations(coeffs, Ts, segment_times, points_per_segment, order=order)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(t_traj, x_acc, 'r-', linewidth=2, label='X Acceleration')
    ax.plot(t_traj, y_acc, 'g-', linewidth=2, label='Y Acceleration')
    ax.plot(t_traj, z_acc, 'b-', linewidth=2, label='Z Acceleration')
    ax.set_ylabel('Acceleration')
    ax.set_xlabel('Time (s)')
    ax.set_title('Acceleration vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    for t in segment_times:
        ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
    return fig, ax, (t_traj, x_acc, y_acc, z_acc)

def plot_jerks(coeffs, Ts, segment_times, points_per_segment=200, order=7):
    """Plot jerk vs time for all three axes"""
    t_traj, x_jerk, y_jerk, z_jerk = get_jerks(coeffs, Ts, segment_times, points_per_segment, order=order)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(t_traj, x_jerk, 'r-', linewidth=2, label='X Jerk')
    ax.plot(t_traj, y_jerk, 'g-', linewidth=2, label='Y Jerk')
    ax.plot(t_traj, z_jerk, 'b-', linewidth=2, label='Z Jerk')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Jerk')
    ax.set_title('Jerk vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    for t in segment_times:
        ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
    return fig, ax, (t_traj, x_jerk, y_jerk, z_jerk)

def plot_snaps(coeffs, Ts, segment_times, points_per_segment=200, order=7):
    """Plot snap vs time for all three axes"""
    t_traj, x_snap, y_snap, z_snap = get_snaps(coeffs, Ts, segment_times, points_per_segment, order=order)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(t_traj, x_snap, 'r-', linewidth=2, label='X Snap')
    ax.plot(t_traj, y_snap, 'g-', linewidth=2, label='Y Snap')
    ax.plot(t_traj, z_snap, 'b-', linewidth=2, label='Z Snap')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Snap')
    ax.set_title('Snap vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    for t in segment_times:
        ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
    return fig, ax, (t_traj, x_snap, y_snap, z_snap)

def compute_forces(t, p_coeffs, order=7):
    """
    Compute motor forces at time t for a given segment using the same method as trajectory_calcs.py
    Supports both 5th and 7th order polynomials.
    """
    # Get trajectory state at time t within the segment
    tau = t  # Time within the segment

    # Get position, velocity, acceleration, jerk using polynomial evaluation
    r = np.array([
        evaluate_polynomial(tau, p_coeffs['x'], order=order),
        evaluate_polynomial(tau, p_coeffs['y'], order=order),
        evaluate_polynomial(tau, p_coeffs['z'], order=order)
    ])

    v = np.array([
        evaluate_velocity(tau, p_coeffs['x'], order=order),
        evaluate_velocity(tau, p_coeffs['y'], order=order),
        evaluate_velocity(tau, p_coeffs['z'], order=order)
    ])

    a = np.array([
        evaluate_acceleration(tau, p_coeffs['x'], order=order),
        evaluate_acceleration(tau, p_coeffs['y'], order=order),
        evaluate_acceleration(tau, p_coeffs['z'], order=order)
    ])

    j = np.array([
        evaluate_jerk(tau, p_coeffs['x'], order=order),
        evaluate_jerk(tau, p_coeffs['y'], order=order),
        evaluate_jerk(tau, p_coeffs['z'], order=order)
    ])

    # Compute snap (4th derivative)
    s = np.array([
        evaluate_snap(tau, p_coeffs['x'], order=order),
        evaluate_snap(tau, p_coeffs['y'], order=order),
        evaluate_snap(tau, p_coeffs['z'], order=order)
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

    return f, a_d_hat

def plot_forces(coeffs, Ts, segment_times, points_per_segment=200, order=7):
    """Plot motor forces vs time, supports order argument"""
    # Get force data using the get function
    t_traj, f1, f2, f3, f4 = get_forces(coeffs, Ts, segment_times, points_per_segment, order=order)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(t_traj, f1, 'r-', linewidth=2, label='Motor 1')
    ax.plot(t_traj, f2, 'g-', linewidth=2, label='Motor 2')
    ax.plot(t_traj, f3, 'b-', linewidth=2, label='Motor 3')
    ax.plot(t_traj, f4, 'orange', linewidth=2, label='Motor 4')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.set_title('Motor Forces vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add vertical lines at segment boundaries
    for t in segment_times:
        ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)

    return fig, ax, (t_traj, f1, f2, f3, f4)

def get_velocities(coeffs, Ts, segment_times, points_per_segment=200, order=7):
    """Get velocity data for all three axes without plotting"""
    # Initialize arrays to store velocity data
    x_vel = []
    y_vel = []
    z_vel = []
    t_traj = []
    
    # Evaluate velocities for each segment
    for i in range(len(Ts)):
        segment_start = segment_times[i]
        
        # Get coefficients for this segment
        p_coeffs = coeffs[f'p{i+1}_coeffs']
        
        # Generate time points for this segment
        t_segment = np.linspace(0, Ts[i], points_per_segment)
        
        # Evaluate velocity for this segment
        x_vel_segment = [evaluate_velocity(t, p_coeffs['x'], order=order) for t in t_segment]
        y_vel_segment = [evaluate_velocity(t, p_coeffs['y'], order=order) for t in t_segment]
        z_vel_segment = [evaluate_velocity(t, p_coeffs['z'], order=order) for t in t_segment]

        # Add to velocity arrays
        x_vel.extend(x_vel_segment)
        y_vel.extend(y_vel_segment)
        z_vel.extend(z_vel_segment)
        t_traj.extend(t_segment + segment_start)
    
    # Convert to numpy arrays
    x_vel = np.array(x_vel)
    y_vel = np.array(y_vel)
    z_vel = np.array(z_vel)
    t_traj = np.array(t_traj)
    
    return t_traj, x_vel, y_vel, z_vel

def get_positions(coeffs, Ts, segment_times, points_per_segment=200, order=7):
    """Get position data for all three axes without plotting"""
    # Initialize arrays to store trajectory points
    x_traj = []
    y_traj = []
    z_traj = []
    t_traj = []
    
    # Evaluate polynomials for each segment
    for i in range(len(Ts)):
        segment_start = segment_times[i]
        segment_end = segment_times[i + 1]
        
        # Get coefficients for this segment
        p_coeffs = coeffs[f'p{i+1}_coeffs']
        
        # Generate time points for this segment
        t_segment = np.linspace(0, Ts[i], points_per_segment)
        
        # Evaluate polynomial for this segment
        x_segment = [evaluate_polynomial(t, p_coeffs['x'], order=order) for t in t_segment]
        y_segment = [evaluate_polynomial(t, p_coeffs['y'], order=order) for t in t_segment]
        z_segment = [evaluate_polynomial(t, p_coeffs['z'], order=order) for t in t_segment]

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
    
    return t_traj, x_traj, y_traj, z_traj

def get_accelerations(coeffs, Ts, segment_times, points_per_segment=200, order=7):
    """Get acceleration data for all three axes without plotting"""
    # Initialize arrays to store acceleration data
    x_acc = []
    y_acc = []
    z_acc = []
    t_traj = []
    
    # Evaluate accelerations for each segment
    for i in range(len(Ts)):
        segment_start = segment_times[i]
        
        # Get coefficients for this segment
        p_coeffs = coeffs[f'p{i+1}_coeffs']
        
        # Generate time points for this segment
        t_segment = np.linspace(0, Ts[i], points_per_segment)
        
        # Evaluate acceleration for this segment
        x_acc_segment = [evaluate_acceleration(t, p_coeffs['x'], order=order) for t in t_segment]
        y_acc_segment = [evaluate_acceleration(t, p_coeffs['y'], order=order) for t in t_segment]
        z_acc_segment = [evaluate_acceleration(t, p_coeffs['z'], order=order) for t in t_segment]

        # Add to acceleration arrays
        x_acc.extend(x_acc_segment)
        y_acc.extend(y_acc_segment)
        z_acc.extend(z_acc_segment)
        t_traj.extend(t_segment + segment_start)
    
    # Convert to numpy arrays
    x_acc = np.array(x_acc)
    y_acc = np.array(y_acc)
    z_acc = np.array(z_acc)
    t_traj = np.array(t_traj)
    
    return t_traj, x_acc, y_acc, z_acc

def get_jerks(coeffs, Ts, segment_times, points_per_segment=200, order=7):
    """Get jerk data for all three axes without plotting"""
    # Initialize arrays to store jerk data
    x_jerk = []
    y_jerk = []
    z_jerk = []
    t_traj = []
    
    # Evaluate jerks for each segment
    for i in range(len(Ts)):
        segment_start = segment_times[i]
        
        # Get coefficients for this segment
        p_coeffs = coeffs[f'p{i+1}_coeffs']
        
        # Generate time points for this segment
        t_segment = np.linspace(0, Ts[i], points_per_segment)
        
        # Evaluate jerk for this segment
        x_jerk_segment = [evaluate_jerk(t, p_coeffs['x'], order=order) for t in t_segment]
        y_jerk_segment = [evaluate_jerk(t, p_coeffs['y'], order=order) for t in t_segment]
        z_jerk_segment = [evaluate_jerk(t, p_coeffs['z'], order=order) for t in t_segment]

        # Add to jerk arrays
        x_jerk.extend(x_jerk_segment)
        y_jerk.extend(y_jerk_segment)
        z_jerk.extend(z_jerk_segment)
        t_traj.extend(t_segment + segment_start)
    
    # Convert to numpy arrays
    x_jerk = np.array(x_jerk)
    y_jerk = np.array(y_jerk)
    z_jerk = np.array(z_jerk)
    t_traj = np.array(t_traj)
    
    return t_traj, x_jerk, y_jerk, z_jerk

def get_snaps(coeffs, Ts, segment_times, points_per_segment=200, order=7):
    """Get snap data for all three axes without plotting"""
    # Initialize arrays to store snap data
    x_snap = []
    y_snap = []
    z_snap = []
    t_traj = []
    
    # Evaluate snaps for each segment
    for i in range(len(Ts)):
        segment_start = segment_times[i]
        
        # Get coefficients for this segment
        p_coeffs = coeffs[f'p{i+1}_coeffs']
        
        # Generate time points for this segment
        t_segment = np.linspace(0, Ts[i], points_per_segment)
        
        # Evaluate snap for this segment
        x_snap_segment = [evaluate_snap(t, p_coeffs['x'], order=order) for t in t_segment]
        y_snap_segment = [evaluate_snap(t, p_coeffs['y'], order=order) for t in t_segment]
        z_snap_segment = [evaluate_snap(t, p_coeffs['z'], order=order) for t in t_segment]

        # Add to snap arrays
        x_snap.extend(x_snap_segment)
        y_snap.extend(y_snap_segment)
        z_snap.extend(z_snap_segment)
        t_traj.extend(t_segment + segment_start)
    
    # Convert to numpy arrays
    x_snap = np.array(x_snap)
    y_snap = np.array(y_snap)
    z_snap = np.array(z_snap)
    t_traj = np.array(t_traj)
    
    return t_traj, x_snap, y_snap, z_snap

def get_forces(coeffs, Ts, segment_times, points_per_segment=200, order=7):
    """Get motor force data without plotting, supports order argument"""
    # Initialize arrays to store force data
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    t_traj = []

    # Compute forces for each segment
    for i in range(len(Ts)):
        segment_start = segment_times[i]

        # Get coefficients for this segment
        p_coeffs = coeffs[f'p{i+1}_coeffs']

        # Generate time points for this segment
        t_segment = np.linspace(0, Ts[i], points_per_segment)

        # Compute forces for this segment
        f1_segment = []
        f2_segment = []
        f3_segment = []
        f4_segment = []

        for t in t_segment:
            forces, _ = compute_forces(t, p_coeffs, order=order)
            f1_segment.append(forces[0])
            f2_segment.append(forces[1])
            f3_segment.append(forces[2])
            f4_segment.append(forces[3])

        # Add to force arrays
        f1.extend(f1_segment)
        f2.extend(f2_segment)
        f3.extend(f3_segment)
        f4.extend(f4_segment)
        t_traj.extend(t_segment + segment_start)

    # Convert to numpy arrays
    f1 = np.array(f1)
    f2 = np.array(f2)
    f3 = np.array(f3)
    f4 = np.array(f4)
    t_traj = np.array(t_traj)

    return t_traj, f1, f2, f3, f4

def print_trajectory_info(coeffs, Ts, total_time, points_per_segment=200, order=7):
    """Print comprehensive trajectory information using polynomial evaluation"""
    # Get trajectory data for analysis
    segment_times = np.cumsum([0] + list(Ts))
    
    # Initialize arrays to store all trajectory data
    x_traj = []
    y_traj = []
    z_traj = []
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
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    
    # Evaluate all data for each segment
    for i in range(len(Ts)):
        segment_start = segment_times[i]
        
        # Get coefficients for this segment
        p_coeffs = coeffs[f'p{i+1}_coeffs']
        
        # Generate time points for this segment
        t_segment = np.linspace(0, Ts[i], points_per_segment)
        
        # Evaluate all quantities for this segment
        for t in t_segment:
            # Position
            x_traj.append(evaluate_polynomial(t, p_coeffs['x'], order=order))
            y_traj.append(evaluate_polynomial(t, p_coeffs['y'], order=order))
            z_traj.append(evaluate_polynomial(t, p_coeffs['z'], order=order))

            # Velocity
            x_vel.append(evaluate_velocity(t, p_coeffs['x'], order=order))
            y_vel.append(evaluate_velocity(t, p_coeffs['y'], order=order))
            z_vel.append(evaluate_velocity(t, p_coeffs['z'], order=order))

            # Acceleration
            x_acc.append(evaluate_acceleration(t, p_coeffs['x'], order=order))
            y_acc.append(evaluate_acceleration(t, p_coeffs['y'], order=order))
            z_acc.append(evaluate_acceleration(t, p_coeffs['z'], order=order))

            # Jerk
            x_jerk.append(evaluate_jerk(t, p_coeffs['x'], order=order))
            y_jerk.append(evaluate_jerk(t, p_coeffs['y'], order=order))
            z_jerk.append(evaluate_jerk(t, p_coeffs['z'], order=order))

            # Snap
            x_snap.append(evaluate_snap(t, p_coeffs['x'], order=order))
            y_snap.append(evaluate_snap(t, p_coeffs['y'], order=order))
            z_snap.append(evaluate_snap(t, p_coeffs['z'], order=order))

            # Forces
            forces = compute_forces(t, p_coeffs, order=order)
            f1.append(forces[0])
            f2.append(forces[1])
            f3.append(forces[2])
            f4.append(forces[3])
    
    # Convert to numpy arrays
    x_traj = np.array(x_traj)
    y_traj = np.array(y_traj)
    z_traj = np.array(z_traj)
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
    f1 = np.array(f1)
    f2 = np.array(f2)
    f3 = np.array(f3)
    f4 = np.array(f4)
    
    # Print information
    print(f"Trajectory Information:")
    print(f"Total duration: {total_time} seconds")
    print(f"Number of segments: {len(Ts)}")
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