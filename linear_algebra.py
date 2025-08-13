# functions to do with solving matrices

from scipy.linalg import solve_banded
import numpy as np
from utils import a, Q_snap, beta

def to_banded(M, l, u):
        """
        Convert a square matrix M to banded form for use with scipy.linalg.solve_banded.
        Args:
            M: Square matrix to convert
            l: Number of lower diagonals
            u: Number of upper diagonals
        Returns:
            M_banded: Banded matrix (shape: (l+u+1, M.shape[1]))
        """
        M_banded = np.zeros((l+u+1, M.shape[1]))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                band_row = u + i - j
                if 0 <= band_row < l+u+1:
                    M_banded[band_row, j] = M[i, j]
        return M_banded

def create_Em(T):
    s = 4
    Em = np.zeros((s, 2*s))
    for i in range(s):
        Em[i,:] = a(i, T)

    return Em

def create_dEmdT(T):
    s = 4
    dEmdT = np.zeros((s, 2*s))
    for i in range(s):
        dEmdT[i,:] = a(i+1, T)

    return dEmdT

def create_F0():
    return create_Em(0)

def create_F():
    s = 4
    F = np.zeros((2*s, 2*s))
    for i in range(2*s-1):
        F[i+1,:] = -a(i, 0)

    return F

def create_E(T):
    s = 4
    E = np.zeros((2*s, 2*s))

    E[0,:] = a(0, T)
    for i in range(2*s-1):
        E[i+1,:] = a(i, T)

    return E

def create_dEdT(T, s=4):
    dEdT = np.zeros((2*s, 2*s))

    dEdT[0,:] = a(1, T)
    for i in range(2*s-1):
        dEdT[i+1,:] = a(1+i, T)

    return dEdT

def create_b(waypoints):
    n = len(waypoints) - 1  # number of segments
    s = 4
    b = np.zeros((2*n*s, 3))
    b[0,:] = waypoints[0]  # Start point of the first segment
    b[-4,:] = waypoints[-1]  # End point of the last segment
    for i in range(n-1):
        b[s*(2*i+1),:] = waypoints[i+1]  # Intermediate waypoints
    return b

def solve_c(M, b, l = 5, u = 3):
    """
    Solve the linear system M * x = b using a banded solver.
    Args:
        M: Coefficient matrix (banded form)
        b: Right-hand side vector
    Returns:
        x: Solution vector
    """

    # Convert M to banded form if necessary
    M_banded = to_banded(M, l, u)
    c = solve_banded((l, u), M_banded, b)

    return c

def create_M(times):
    n = len(times) # number of segments
    s = 4
    # Create a block-diagonal matrix M for the constraints
    M = np.zeros((2*n*s, 2*n*s))
    M[:s, :2*s] = create_F0()
    F = create_F()
    for i in range(n-1):
        T = times[i]
        M[s*(2*i+1):s*(2*i+3), s*(2*i):s*(2*i+2)] = create_E(T)
        M[s*(2*i+1):s*(2*i+3), s*(2*i+2):s*(2*i+4)] = F
    M[s*(2*n-1):, s*(2*n-2):] = create_Em(times[-1])
    
    return M

# construct A matrix for min-snap
def calc_A(times):
    """
    Construct the constraint matrix A for minimum snap trajectory.
    Args:
        times: list or array of segment times (N-1)
    Returns:
        A: constraint matrix (shape: (5*n+3+3*(n-1), s*n, where s=8, n=#segments))
    """
    s = 8
    n = len(times) 
    A = np.zeros((5*n+3+3*(n-1), s*n))
    # position constraints
    for i in range(n):
        t_left = 0
        t_right = times[i] 
        A[2*i, i*s:(i+1)*s] = a(0, t_left)  # Start of segment i
        A[2*i+1, i*s:(i+1)*s] = a(0, t_right)  # End of segment i

    # hovering at 1st and last segment
    for i in range(3):
        A[2*n+i, 0:s] = a(i+1, 0)
        A[2*n+3+i, (n-1)*s:n*s] = a(i+1, times[-1])

    # continuity constraints for velocity, acceleration, jerk at intermediate waypoints
    for i in range(n-1):
        A[2*n+6+3*i, i*s:(i+1)*s] = a(1, times[i])  # Velocity continuity from the left
        A[2*n+6+3*i, (i+1)*s:(i+2)*s] = -a(1, 0) # Velocity continuity from the right
        A[2*n+6+3*i+1, i*s:(i+1)*s] = a(2, times[i])
        A[2*n+6+3*i+1, (i+1)*s:(i+2)*s] = -a(2, 0)
        A[2*n+6+3*i+2, i*s:(i+1)*s] = a(3, times[i])
        A[2*n+6+3*i+2, (i+1)*s:(i+2)*s] = -a(3, 0)

    # the rest are free variables 
    # there are 3 * (n-1)
    for i in range(n-1):
        A[5*n+3+3*i, i*s:(i+1)*s] = a(1, times[i])
        A[5*n+3+3*i+1, i*s:(i+1)*s] = a(2, times[i])
        A[5*n+3+3*i+2, i*s:(i+1)*s] = a(3, times[i])    
    return A

def grad_Q(T):

    gradQ = np.zeros((8, 8))
    for i in range(4,8):
        for j in range(4,8):
            gradQ[i,j] = beta(i) * beta(j) * T**(i+j-8)

    return gradQ

# Construct the block-diagonal Q matrix for min-snap
def calc_Q(times):
    s = 8
    n = len(times)
    Q = np.zeros((s * n, s * n))
    for i in range(n):
        T = times[i]
        Q_i = Q_snap(T)
        Q[i*s:(i+1)*s, i*s:(i+1)*s] = Q_i
    return Q

def calc_dKdT(c, T, kT):
    """
    Calculate the derivative of the cost function with respect to time T.
    Args:
        c: Coefficients vector
        T: Current time segment
        kT: Temperature parameter
    Returns:
        dKdT: Derivative of the cost function with respect to T
    """
    gradQ = grad_Q(T)
    dKdT = np.sum(c.T @ gradQ @ c) + kT  # Add time penalty term
    return dKdT

def calc_dKdc_1(c, Q):
    """
    This is preferred because Q is provided.
    Calculate the derivative of the cost function with respect to coefficients c.
    Args:
        c: Coefficients vector
        Q: Quadratic cost matrix
    Returns:
        dKdc: Derivative of the cost function with respect to c
    """

    dKdc = 2 * Q @ c  # Gradient of the quadratic form
    return dKdc

def calc_dKdc_2(c, times):
    """
    Calculate the derivative of the cost function with respect to coefficients c.
    Args:
        c: Coefficients vector
        times: Current time segments
    Returns:
        dKdc: Derivative of the cost function with respect to c
    """
    dKdc = 2 * calc_Q(times) @ c  # Gradient of the quadratic form
    return dKdc

def calc_G(M, dKdc, l = 5, u = 3):
    A = to_banded(M.T, u, l)
    return solve_banded((u, l), A, dKdc)

def calc_dWdT(c, T, kT, G_partition, dEdT):
    dKdT = calc_dKdT(c, T, kT)
    dWdT = dKdT - np.trace(G_partition.T @ dEdT @ c)
    return dWdT

if __name__ == "__main__":
    # Example usage
    times = [1.0, 2.0, 1.5]  # Example segment times
    waypoints = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 2], [3, 3, 3]])  # Example waypoints
    M = create_M(times)
    b = create_b(waypoints)
    c = solve_c(M, b)

    i = 0
    coeffs = c[i*8:(i+1)*8, :]  # Extract coefficients for segment i
    T = times[i]  # Current time segment

    print(Q_snap(T).shape)
    print(coeffs.shape)
    
    dKdc = calc_dKdc_2(coeffs, T)