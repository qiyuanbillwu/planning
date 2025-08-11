# functions to do with solving matrices

from scipy.linalg import solve_banded
import numpy as np
from utils import a

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
