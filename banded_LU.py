import numpy as np
from scipy.linalg import lu, solve_banded, lu_solve
import time

def generate_random_banded(n, l, u):
    """
    Generate a random banded matrix of size n x n with l lower and u upper bands.
    """
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i-l), min(n, i+u+1)):
            A[i, j] = np.random.randn()
    return A


def to_banded(A, l, u):
    """
    Convert full matrix A to banded form for solve_banded.
    Returns ab with shape (l+u+1, n)
    """
    n = A.shape[0]
    ab = np.zeros((l+u+1, n))
    for i in range(n):
        for j in range(max(0, i-l), min(n, i+u+1)):
            ab[u + i - j, j] = A[i, j]
    return ab

# Example usage:
if __name__ == "__main__":
    n, l, u = 100, 2, 2
    N = 100  # Number of repetitions for averaging

    banded_times = []
    lu_times = []
    max_diffs = []

    for _ in range(N):
        A = generate_random_banded(n, l, u)
        b = np.random.randn(n)

        # --- solve_banded ---
        ab = to_banded(A, l, u)
        t0 = time.time()
        x_banded = solve_banded((l, u), ab, b)
        t1 = time.time()
        banded_times.append(t1 - t0)

        # --- LU decomposition + solve ---
        t2 = time.time()
        P, L, U = lu(A)
        y = np.linalg.solve(L, P @ b)
        x_lu = np.linalg.solve(U, y)
        t3 = time.time()
        lu_times.append(t3 - t2)

        max_diffs.append(np.max(np.abs(x_banded - x_lu)))

    print(f"Average solve_banded time over {N} runs: {np.mean(banded_times):.6f} seconds")
    print(f"Average LU decomposition + solve time over {N} runs: {np.mean(lu_times):.6f} seconds")
    print(f"Average max difference in solution: {np.mean(max_diffs):.2e}")

