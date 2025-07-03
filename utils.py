import numpy as np

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
    else:
        raise ValueError("n must be between 0 and 4 inclusive")

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

def Q(T):
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

# New cost matrix for 5th order (min-jerk) polynomial
# This is for minimum jerk trajectory generation, integrating the square of the 3rd derivative (jerk)
def Q5(T):
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