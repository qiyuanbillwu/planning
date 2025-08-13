from webbrowser import get
from utils import a, a5, Q_snap, Q_jerk, beta
from scipy.linalg import solve_banded
from linear_algebra import *
import constants
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from waypoints import waypoint_list

# Extract 8 waypoints from waypoints.txt
# r0 = np.array([18.2908, -12.9164, 0.5])
# r1 = np.array([16.0048, -6.01777, 0.746351])
# r2 = np.array([9.74278, -4.28989, 3.58934])
# r3 = np.array([2.32316, -1.06404, 1.57101])
# r4 = np.array([-2.50561, 5.7747, 1.74195])
# r5 = np.array([-5.96077, 10.9205, 1.32572])
# r6 = np.array([-16.5275, 15.9659, 1.26184])
# r7 = np.array([-19.8453, 12.2357, 0.5])

r0 = np.array([1, 1, 0.5])
r1 = np.array([0, 1.5, 1])
r2 = np.array([-0.8, 1, 2])
r3 = np.array([-1.5, 0, 2.5])
r4 = np.array([-1, -1, 1.5])
r5 = np.array([0.25, -0.75, 1])
r6 = np.array([1, 0, 1.5])
r7 = np.array([0, 0, 2])

def calc_time(waypoint_list, v_avg=constants.v_avg):
    # Calculate the time taken to traverse the waypoints
    distances = [np.linalg.norm(waypoint_list[i+1] - waypoint_list[i]) for i in range(len(waypoint_list)-1)]
    return np.array(distances) / v_avg

def get_coefficients(Ts, save=True):

    T1 = Ts[0]
    T2 = Ts[1]
    T3 = Ts[2]
    T4 = Ts[3]
    T5 = Ts[4]
    T6 = Ts[5]
    T7 = Ts[6]

    Q1 = Q_jerk(T1)
    Q2 = Q_jerk(T2)
    Q3 = Q_jerk(T3)
    Q4 = Q_jerk(T4)
    Q5 = Q_jerk(T5)
    Q6 = Q_jerk(T6)
    Q7 = Q_jerk(T7)
    # Q is a 6x6 matrix for a 5th order (min-jerk) polynomial

    # Create an empty (6*7, 6*7) matrix and assign Q1 to Q7 to the diagonal blocks
    Q_total = np.zeros((6 * 7, 6 * 7))
    Q_list = [Q1, Q2, Q3, Q4, Q5, Q6, Q7]
    for i, Q in enumerate(Q_list):
        Q_total[i*6:(i+1)*6, i*6:(i+1)*6] = Q

    df = np.block([
        # waypoints
        [r0],
        [r1],
        [r1],
        [r2],
        [r2],
        [r3],
        [r3],
        [r4],
        [r4],   
        [r5],
        [r5],
        [r6],
        [r6],
        [r7],
        # continuity constraints on velocity and acceleration
        [np.zeros((1, 3))], # v1
        [np.zeros((1, 3))], # a1
        [np.zeros((1, 3))], # v2
        [np.zeros((1, 3))], # a2
        [np.zeros((1, 3))], # v3
        [np.zeros((1, 3))], # a3
        [np.zeros((1, 3))], # v4
        [np.zeros((1, 3))], # a4
        [np.zeros((1, 3))], # v5
        [np.zeros((1, 3))], # a5
        [np.zeros((1, 3))], # v6
        [np.zeros((1, 3))], # a6
    ])

    A = np.block([
        [a5(0,0), np.zeros((1, 6*6))], # r0 (start of seg 1)
        [a5(0,T1), np.zeros((1, 6*6))], # r1 (end of seg 1)
        [np.zeros((1, 6)), a5(0,0), np.zeros((1, 6*5))], # r1 (start of seg 2)
        [np.zeros((1, 6)), a5(0, T2), np.zeros((1, 6*5))], # r2 (end of seg 2)
        [np.zeros((1, 6*2)), a5(0,0), np.zeros((1, 6*4))], # r2 (start of seg 3)
        [np.zeros((1, 6*2)), a5(0, T3), np.zeros((1, 6*4))], # r3 (end of seg 3)
        [np.zeros((1, 6*3)), a5(0,0), np.zeros((1, 6*3))], # r3 (start of seg 4)
        [np.zeros((1, 6*3)), a5(0, T4), np.zeros((1, 6*3))], # r4 (end of seg 4)
        [np.zeros((1, 6*4)), a5(0,0), np.zeros((1, 6*2))], # r4 (start of seg 5)
        [np.zeros((1, 6*4)), a5(0, T5), np.zeros((1, 6*2))], # r5 (end of seg 5)
        [np.zeros((1, 6*5)), a5(0,0), np.zeros((1, 6*1))], # r5 (start of seg 6)
        [np.zeros((1, 6*5)), a5(0, T6), np.zeros((1, 6*1))], # r6 (end of seg 6)
        [np.zeros((1, 6*6)), a5(0,0)], # r6 (start of seg 7)
        [np.zeros((1, 6*6)), a5(0, T7)], # r7 (end of seg 7)
        [a5(1,T1), -a5(1,0), np.zeros((1, 6*5))], # v1
        [a5(2,T1), -a5(2,0), np.zeros((1, 6*5))], # a1
        [np.zeros((1, 6)), a5(1, T2), -a5(1, 0), np.zeros((1, 6*4))], # v2
        [np.zeros((1, 6)), a5(2, T2), -a5(2, 0), np.zeros((1, 6*4))], # a2
        [np.zeros((1, 6*2)), a5(1, T3), -a5(1, 0), np.zeros((1, 6*3))], # v3
        [np.zeros((1, 6*2)), a5(2, T3), -a5(2, 0), np.zeros((1, 6*3))], # a3
        [np.zeros((1, 6*3)), a5(1, T4), -a5(1, 0), np.zeros((1, 6*2))], # v4
        [np.zeros((1, 6*3)), a5(2, T4), -a5(2, 0), np.zeros((1, 6*2))], # a4
        [np.zeros((1, 6*4)), a5(1, T5), -a5(1, 0), np.zeros((1, 6*1))], # v5
        [np.zeros((1, 6*4)), a5(2, T5), -a5(2, 0), np.zeros((1, 6*1))], # a5 
        [np.zeros((1, 6*5)), a5(1, T6), -a5(1, 0)], # v6
        [np.zeros((1, 6*5)), a5(2, T6), -a5(2, 0)], # a6
        [a5(1,0), np.zeros((1, 6*6))], # v0
        [a5(2,0), np.zeros((1, 6*6))], # a0
        [a5(1,T1), np.zeros((1, 6*6))], # v1
        [a5(2,T1), np.zeros((1, 6*6))], # a1
        [np.zeros((1, 6)), a5(1,T2), np.zeros((1, 6*5))], # v2
        [np.zeros((1, 6)), a5(2,T2), np.zeros((1, 6*5))], # a2
        [np.zeros((1, 6*2)), a5(1,T3), np.zeros((1, 6*4))], # v3
        [np.zeros((1, 6*2)), a5(2,T3), np.zeros((1, 6*4))], # a3
        [np.zeros((1, 6*3)), a5(1,T4), np.zeros((1, 6*3))], # v4
        [np.zeros((1, 6*3)), a5(2,T4), np.zeros((1, 6*3))], # a4
        [np.zeros((1, 6*4)), a5(1,T5), np.zeros((1, 6*2))], # v5
        [np.zeros((1, 6*4)), a5(2,T5), np.zeros((1, 6*2))], # a5
        [np.zeros((1, 6*5)), a5(1,T6), np.zeros((1, 6*1))], # v6
        [np.zeros((1, 6*5)), a5(2,T6), np.zeros((1, 6*1))], # a6
        [np.zeros((1, 6*6)), a5(1,T7)], # v7
        [np.zeros((1, 6*6)), a5(2,T7)], # a7
    ])

    # Compute the inverse of A
    A_inv = np.linalg.inv(A)

    R = A_inv.T @ Q_total @ A_inv

    # Partition R based on the sizes of df and dp
    n_f = df.shape[0]  # number of fixed variables
    n_p = 6 * 7 - n_f  # number of free variables

    # print(n_f, n_p)

    R_FF = R[:n_f, :n_f]
    R_FP = R[:n_f, n_f:n_f+n_p]
    R_PF = R[n_f:n_f+n_p, :n_f]
    R_PP = R[n_f:n_f+n_p, n_f:n_f+n_p]

    dp_star = -np.linalg.inv(R_PP) @ R_FP.T @ df

    # Assemble the complete solution vector d from df and dp_star
    d = np.vstack([df, dp_star])

    p = A_inv @ d

    if save == True:
        # Extract coefficients for each polynomial segment (x, y, z coordinates)
        p1_coeffs = p[:6, :]  # First 6 coefficients for polynomial 1 (x, y, z)
        p2_coeffs = p[6:12, :]  # Next 6 coefficients for polynomial 2 (x, y, z)
        p3_coeffs = p[12:18, :]  # Next 6 coefficients for polynomial 3 (x, y, z)
        p4_coeffs = p[18:24, :]  # Next 6 coefficients for polynomial 4 (x, y, z)
        p5_coeffs = p[24:30, :]  # Next 6 coefficients for polynomial 5 (x, y, z)
        p6_coeffs = p[30:36, :]  # Next 6 coefficients for polynomial 6 (x, y, z)
        p7_coeffs = p[36:42, :]  # Last 6 coefficients for polynomial 7 (x, y, z)

        # Create data directory if it doesn't exist
        import os
        if not os.path.exists('data'):
            os.makedirs('data')

        # Save coefficients to JSON file
        import json
        coefficients = {
            "p1_coeffs": {
                "x": p1_coeffs[:, 0].tolist(),
                "y": p1_coeffs[:, 1].tolist(),
                "z": p1_coeffs[:, 2].tolist()
            },
            "p2_coeffs": {
                "x": p2_coeffs[:, 0].tolist(),
                "y": p2_coeffs[:, 1].tolist(),
                "z": p2_coeffs[:, 2].tolist()
            },
            "p3_coeffs": {
                "x": p3_coeffs[:, 0].tolist(),
                "y": p3_coeffs[:, 1].tolist(),
                "z": p3_coeffs[:, 2].tolist()
            },
            "p4_coeffs": {
                "x": p4_coeffs[:, 0].tolist(),
                "y": p4_coeffs[:, 1].tolist(),
                "z": p4_coeffs[:, 2].tolist()
            },
            "p5_coeffs": {
                "x": p5_coeffs[:, 0].tolist(),
                "y": p5_coeffs[:, 1].tolist(),
                "z": p5_coeffs[:, 2].tolist()
            },
            "p6_coeffs": {
                "x": p6_coeffs[:, 0].tolist(),
                "y": p6_coeffs[:, 1].tolist(),
                "z": p6_coeffs[:, 2].tolist()
            },
            "p7_coeffs": {
                "x": p7_coeffs[:, 0].tolist(),
                "y": p7_coeffs[:, 1].tolist(),
                "z": p7_coeffs[:, 2].tolist()
            }
        }
        
        with open('data/polynomial_coefficients.json', 'w') as f:
            json.dump(coefficients, f, indent=2)

    return p

def J(Ts, kT):

    T1 = Ts[0]
    T2 = Ts[1]
    T3 = Ts[2]
    T4 = Ts[3]
    T5 = Ts[4]
    T6 = Ts[5]
    T7 = Ts[6]

    Q1 = Q_jerk(T1)
    Q2 = Q_jerk(T2)
    Q3 = Q_jerk(T3)
    Q4 = Q_jerk(T4)
    Q5 = Q_jerk(T5)
    Q6 = Q_jerk(T6)
    Q7 = Q_jerk(T7)
    # Q is a 6x6 matrix for a 5th order (min-jerk) polynomial

    # Create an empty (6*7, 6*7) matrix and assign Q1 to Q7 to the diagonal blocks
    Q_total = np.zeros((6 * 7, 6 * 7))
    Q_list = [Q1, Q2, Q3, Q4, Q5, Q6, Q7]
    for i, Q in enumerate(Q_list):
        Q_total[i*6:(i+1)*6, i*6:(i+1)*6] = Q

    df = np.block([
        # waypoints
        [r0],
        [r1],
        [r1],
        [r2],
        [r2],
        [r3],
        [r3],
        [r4],
        [r4],   
        [r5],
        [r5],
        [r6],
        [r6],
        [r7],
        # continuity constraints on velocity and acceleration
        [np.zeros((1, 3))], # v1
        [np.zeros((1, 3))], # a1
        [np.zeros((1, 3))], # v2
        [np.zeros((1, 3))], # a2
        [np.zeros((1, 3))], # v3
        [np.zeros((1, 3))], # a3
        [np.zeros((1, 3))], # v4
        [np.zeros((1, 3))], # a4
        [np.zeros((1, 3))], # v5
        [np.zeros((1, 3))], # a5
        [np.zeros((1, 3))], # v6
        [np.zeros((1, 3))], # a6
    ])

    A = np.block([
        [a5(0,0), np.zeros((1, 6*6))], # r0 (start of seg 1)
        [a5(0,T1), np.zeros((1, 6*6))], # r1 (end of seg 1)
        [np.zeros((1, 6)), a5(0,0), np.zeros((1, 6*5))], # r1 (start of seg 2)
        [np.zeros((1, 6)), a5(0, T2), np.zeros((1, 6*5))], # r2 (end of seg 2)
        [np.zeros((1, 6*2)), a5(0,0), np.zeros((1, 6*4))], # r2 (start of seg 3)
        [np.zeros((1, 6*2)), a5(0, T3), np.zeros((1, 6*4))], # r3 (end of seg 3)
        [np.zeros((1, 6*3)), a5(0,0), np.zeros((1, 6*3))], # r3 (start of seg 4)
        [np.zeros((1, 6*3)), a5(0, T4), np.zeros((1, 6*3))], # r4 (end of seg 4)
        [np.zeros((1, 6*4)), a5(0,0), np.zeros((1, 6*2))], # r4 (start of seg 5)
        [np.zeros((1, 6*4)), a5(0, T5), np.zeros((1, 6*2))], # r5 (end of seg 5)
        [np.zeros((1, 6*5)), a5(0,0), np.zeros((1, 6*1))], # r5 (start of seg 6)
        [np.zeros((1, 6*5)), a5(0, T6), np.zeros((1, 6*1))], # r6 (end of seg 6)
        [np.zeros((1, 6*6)), a5(0,0)], # r6 (start of seg 7)
        [np.zeros((1, 6*6)), a5(0, T7)], # r7 (end of seg 7)
        [a5(1,T1), -a5(1,0), np.zeros((1, 6*5))], # v1
        [a5(2,T1), -a5(2,0), np.zeros((1, 6*5))], # a1
        [np.zeros((1, 6)), a5(1, T2), -a5(1, 0), np.zeros((1, 6*4))], # v2
        [np.zeros((1, 6)), a5(2, T2), -a5(2, 0), np.zeros((1, 6*4))], # a2
        [np.zeros((1, 6*2)), a5(1, T3), -a5(1, 0), np.zeros((1, 6*3))], # v3
        [np.zeros((1, 6*2)), a5(2, T3), -a5(2, 0), np.zeros((1, 6*3))], # a3
        [np.zeros((1, 6*3)), a5(1, T4), -a5(1, 0), np.zeros((1, 6*2))], # v4
        [np.zeros((1, 6*3)), a5(2, T4), -a5(2, 0), np.zeros((1, 6*2))], # a4
        [np.zeros((1, 6*4)), a5(1, T5), -a5(1, 0), np.zeros((1, 6*1))], # v5
        [np.zeros((1, 6*4)), a5(2, T5), -a5(2, 0), np.zeros((1, 6*1))], # a5 
        [np.zeros((1, 6*5)), a5(1, T6), -a5(1, 0)], # v6
        [np.zeros((1, 6*5)), a5(2, T6), -a5(2, 0)], # a6
        [a5(1,0), np.zeros((1, 6*6))], # v0
        [a5(2,0), np.zeros((1, 6*6))], # a0
        [a5(1,T1), np.zeros((1, 6*6))], # v1
        [a5(2,T1), np.zeros((1, 6*6))], # a1
        [np.zeros((1, 6)), a5(1,T2), np.zeros((1, 6*5))], # v2
        [np.zeros((1, 6)), a5(2,T2), np.zeros((1, 6*5))], # a2
        [np.zeros((1, 6*2)), a5(1,T3), np.zeros((1, 6*4))], # v3
        [np.zeros((1, 6*2)), a5(2,T3), np.zeros((1, 6*4))], # a3
        [np.zeros((1, 6*3)), a5(1,T4), np.zeros((1, 6*3))], # v4
        [np.zeros((1, 6*3)), a5(2,T4), np.zeros((1, 6*3))], # a4
        [np.zeros((1, 6*4)), a5(1,T5), np.zeros((1, 6*2))], # v5
        [np.zeros((1, 6*4)), a5(2,T5), np.zeros((1, 6*2))], # a5
        [np.zeros((1, 6*5)), a5(1,T6), np.zeros((1, 6*1))], # v6
        [np.zeros((1, 6*5)), a5(2,T6), np.zeros((1, 6*1))], # a6
        [np.zeros((1, 6*6)), a5(1,T7)], # v7
        [np.zeros((1, 6*6)), a5(2,T7)], # a7
    ])

    # Compute the inverse of A
    A_inv = np.linalg.inv(A)

    R = A_inv.T @ Q_total @ A_inv

    # Partition R based on the sizes of df and dp
    n_f = df.shape[0]  # number of fixed variables
    n_p = 6 * 7 - n_f  # number of free variables

    # print(n_f, n_p)

    R_FF = R[:n_f, :n_f]
    R_FP = R[:n_f, n_f:n_f+n_p]
    R_PF = R[n_f:n_f+n_p, :n_f]
    R_PP = R[n_f:n_f+n_p, n_f:n_f+n_p]

    dp_star = -np.linalg.inv(R_PP) @ R_FP.T @ df

    # Assemble the complete solution vector d from df and dp_star
    d = np.vstack([df, dp_star])

    p = A_inv @ d

    # Extract coefficients for each polynomial segment (x, y, z coordinates)
    p1_coeffs = p[:6, :]  # First 6 coefficients for polynomial 1 (x, y, z)
    p2_coeffs = p[6:12, :]  # Next 6 coefficients for polynomial 2 (x, y, z)
    p3_coeffs = p[12:18, :]  # Next 6 coefficients for polynomial 3 (x, y, z)
    p4_coeffs = p[18:24, :]  # Next 6 coefficients for polynomial 4 (x, y, z)
    p5_coeffs = p[24:30, :]  # Next 6 coefficients for polynomial 5 (x, y, z)
    p6_coeffs = p[30:36, :]  # Next 6 coefficients for polynomial 6 (x, y, z)
    p7_coeffs = p[36:42, :]  # Last 6 coefficients for polynomial 7 (x, y, z)

    J = (p1_coeffs[:,0] @ Q1 @ p1_coeffs[:,0].T + p1_coeffs[:,1] @ Q1 @ p1_coeffs[:,1].T + p1_coeffs[:,2] @ Q1 @ p1_coeffs[:,2].T +
         p2_coeffs[:,0] @ Q2 @ p2_coeffs[:,0].T + p2_coeffs[:,1] @ Q2 @ p2_coeffs[:,1].T + p2_coeffs[:,2] @ Q2 @ p2_coeffs[:,2].T +
         p3_coeffs[:,0] @ Q3 @ p3_coeffs[:,0].T + p3_coeffs[:,1] @ Q3 @ p3_coeffs[:,1].T + p3_coeffs[:,2] @ Q3 @ p3_coeffs[:,2].T +
         p4_coeffs[:,0] @ Q4 @ p4_coeffs[:,0].T + p4_coeffs[:,1] @ Q4 @ p4_coeffs[:,1].T + p4_coeffs[:,2] @ Q4 @ p4_coeffs[:,2].T +
         p5_coeffs[:,0] @ Q5 @ p5_coeffs[:,0].T + p5_coeffs[:,1] @ Q5 @ p5_coeffs[:,1].T + p5_coeffs[:,2] @ Q5 @ p5_coeffs[:,2].T +
         p6_coeffs[:,0] @ Q6 @ p6_coeffs[:,0].T + p6_coeffs[:,1] @ Q6 @ p6_coeffs[:,1].T + p6_coeffs[:,2] @ Q6 @ p6_coeffs[:,2].T +
         p7_coeffs[:,0] @ Q7 @ p7_coeffs[:,0].T + p7_coeffs[:,1] @ Q7 @ p7_coeffs[:,1].T + p7_coeffs[:,2] @ Q7 @ p7_coeffs[:,2].T +
         kT * (T1 + T2 + T3 + T4 + T5 + T6 + T7))

    return J

def optimize_times_gradient_descent(Ts_initial, kT, learning_rate=1e-4, max_iterations=1000, tolerance=1e-4):
    """
    Optimize time parameters using gradient descent.
    
    Args:
        Ts_initial: Initial time parameters array
        learning_rate: Learning rate for gradient descent
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        kT: Time penalty coefficient
    
    Returns:
        dict: Dictionary containing optimization results
    """
    # Store history for plotting
    Ts_history = [Ts_initial.copy()]
    cost_history = []
    
    print("Starting gradient descent optimization...")
    
    # Start timing
    start_time_gd = time.time()
    
    Ts_current = Ts_initial.copy()
    
    for iteration in range(max_iterations):
        # Calculate current cost
        current_cost = J(Ts_current, kT)
        cost_history.append(current_cost)
        
        # Calculate gradients using finite differences
        delta = 1e-8  # Increased for stability and speed
        grad_Ts = np.zeros_like(Ts_current)
        
        for i in range(len(Ts_current)):
            Ts_plus = Ts_current.copy()
            Ts_plus[i] += delta
            cost_plus = J(Ts_plus, kT)
            grad_Ts[i] = (cost_plus - current_cost) / delta
        
        # Update Ts
        Ts_new = Ts_current - learning_rate * grad_Ts
        
        # Ensure all Ts remain positive
        Ts_new = np.maximum(0.1, Ts_new)
        
        # Check convergence
        Ts_change = np.max(np.abs(Ts_new - Ts_current))

        # Print progress every 100 iterations and always print the first and last
        if iteration % 100 == 0 or iteration == 0 or iteration == max_iterations - 1:
            print(f"Iteration {iteration}: Cost={current_cost:.6f}, Ts={Ts_current}, Max ΔTs={Ts_change:.6e}")
        
        # Update current values
        Ts_current = Ts_new
        Ts_history.append(Ts_current.copy())
        
        # Check for convergence
        if Ts_change < tolerance:
            print(f"Converged after  {iteration} iterations")
            break
    
    end_time_gd = time.time()
    optimization_time = end_time_gd - start_time_gd
    
    print(f"Optimization completed in {optimization_time:.2f} seconds")
    print(f"Final Ts: {Ts_current}")
    print(f"Final cost: {current_cost:.6f}")
    
    # Calculate total time and proportions
    total_time = np.sum(Ts_current)
    time_proportions = Ts_current / total_time
    
    # Return results as dictionary
    results = {
        'Ts_optimized': Ts_current,
        'final_cost': current_cost,
        'total_time': total_time,
        'time_proportions': time_proportions,
        'Ts_history': np.array(Ts_history),
        'cost_history': cost_history,
        'optimization_time_seconds': optimization_time,
        'iterations': len(cost_history),
        'converged': bool(Ts_change < tolerance)
    }
    
    return results

def plot_optimization_history(results):
    """
    Plot the optimization history.
    
    Args:
        results: Dictionary returned by optimize_times_gradient_descent
    """
    plt.figure(figsize=(12, 8))
    
    # Plot cost history
    plt.subplot(2, 1, 1)
    plt.plot(results['cost_history'])
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function History')
    plt.grid(True)
    
    # Plot Ts history
    plt.subplot(2, 1, 2)
    Ts_history_array = results['Ts_history']
    for i in range(len(results['Ts_optimized'])):
        plt.plot(Ts_history_array[:, i], label=f'T{i+1}')
    plt.xlabel('Iteration')
    plt.ylabel('Time (s)')
    plt.title('Time Parameters History')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    # plt.show()

def save_optimization_results(results, coefficients):
    """
    Save optimization results to files.
    
    Args:
        results: Dictionary returned by optimize_times_gradient_descent
        coefficients: Coefficients returned by get_coefficients
    """
    # Save time parameters to JSON file
    time_data = {
        "optimized_times": results['Ts_optimized'].tolist(),
        "total_time": results['total_time'],
        "time_proportions": results['time_proportions'].tolist(),
        "final_cost": results['final_cost'],
        "optimization_iterations": results['iterations'],
        "optimization_time_seconds": results['optimization_time_seconds'],
        "converged": results['converged']
    }
    
    with open('data/optimized_time_parameters.json', 'w') as f:
        json.dump(time_data, f, indent=2)
    
    # print("Optimized parameters saved to:")
    # print("- data/polynomial_coefficients.json (coefficients)")
    # print("- data/optimized_time_parameters.json (time parameters)")

def get_coefficients_min_snap(Ts, save=True):
    """
    Compute polynomial coefficients for minimum snap (7th order, 8 coefficients per segment).
    Structure and output matches get_coefficients (min-jerk) but uses min-snap constraints.
    """
    T1, T2, T3, T4, T5, T6, T7 = Ts

    Q1 = Q_snap(T1)
    Q2 = Q_snap(T2)
    Q3 = Q_snap(T3)
    Q4 = Q_snap(T4)
    Q5 = Q_snap(T5)
    Q6 = Q_snap(T6)
    Q7 = Q_snap(T7)
    # Q is an 8x8 matrix for a 7th order (min-snap) polynomial

    # Create an empty (8*7, 8*7) matrix and assign Q1 to Q7 to the diagonal blocks
    Q_total = np.zeros((8 * 7, 8 * 7))
    Q_list = [Q1, Q2, Q3, Q4, Q5, Q6, Q7]
    for i, Q in enumerate(Q_list):
        Q_total[i*8:(i+1)*8, i*8:(i+1)*8] = Q

    # Waypoints
    waypoints = [r0, r1, r2, r3, r4, r5, r6, r7]

    # Build df: positions at segment endpoints, zeros for continuity constraints (vel, acc, jerk, snap)
    df = np.block([
        [r0],
        [r1],
        [r1],
        [r2],
        [r2],
        [r3],
        [r3],
        [r4],
        [r4],
        [r5],
        [r5],
        [r6],
        [r6],
        [r7],
        # continuity constraints (vel, acc, jerk, snap) at each internal waypoint
        [np.zeros((1, 3))], # v1
        [np.zeros((1, 3))], # a1
        [np.zeros((1, 3))], # j1
        [np.zeros((1, 3))], # s1
        [np.zeros((1, 3))], # v2
        [np.zeros((1, 3))], # a2
        [np.zeros((1, 3))], # j2
        [np.zeros((1, 3))], # s2
        [np.zeros((1, 3))], # v3
        [np.zeros((1, 3))], # a3
        [np.zeros((1, 3))], # j3
        [np.zeros((1, 3))], # s3
        [np.zeros((1, 3))], # v4
        [np.zeros((1, 3))], # a4
        [np.zeros((1, 3))], # j4
        [np.zeros((1, 3))], # s4
        [np.zeros((1, 3))], # v5
        [np.zeros((1, 3))], # a5
        [np.zeros((1, 3))], # j5
        [np.zeros((1, 3))], # s5
        [np.zeros((1, 3))], # v6
        [np.zeros((1, 3))], # a6
        [np.zeros((1, 3))], # j6
        [np.zeros((1, 3))], # s6
    ])

    # Build A matrix using a (from utils) for 7th order polynomials
    A = np.block([
        [a(0,0), np.zeros((1, 8*6))], # r0 (start of seg 1)
        [a(0,T1), np.zeros((1, 8*6))], # r1 (end of seg 1)
        [np.zeros((1, 8)), a(0,0), np.zeros((1, 8*5))], # r1 (start of seg 2)
        [np.zeros((1, 8)), a(0,T2), np.zeros((1, 8*5))], # r2 (end of seg 2)
        [np.zeros((1, 8*2)), a(0,0), np.zeros((1, 8*4))], # r2 (start of seg 3)
        [np.zeros((1, 8*2)), a(0,T3), np.zeros((1, 8*4))], # r3 (end of seg 3)
        [np.zeros((1, 8*3)), a(0,0), np.zeros((1, 8*3))], # r3 (start of seg 4)
        [np.zeros((1, 8*3)), a(0,T4), np.zeros((1, 8*3))], # r4 (end of seg 4)
        [np.zeros((1, 8*4)), a(0,0), np.zeros((1, 8*2))], # r4 (start of seg 5)
        [np.zeros((1, 8*4)), a(0,T5), np.zeros((1, 8*2))], # r5 (end of seg 5)
        [np.zeros((1, 8*5)), a(0,0), np.zeros((1, 8*1))], # r5 (start of seg 6)
        [np.zeros((1, 8*5)), a(0,T6), np.zeros((1, 8*1))], # r6 (end of seg 6)
        [np.zeros((1, 8*6)), a(0,0)], # r6 (start of seg 7)
        [np.zeros((1, 8*6)), a(0,T7)], # r7 (end of seg 7)
        # velocity, acceleration, jerk, snap continuity at each internal waypoint
        [a(1,T1), -a(1,0), np.zeros((1, 8*5))], # v1
        [a(2,T1), -a(2,0), np.zeros((1, 8*5))], # a1
        [a(3,T1), -a(3,0), np.zeros((1, 8*5))], # j1
        [a(4,T1), -a(4,0), np.zeros((1, 8*5))], # s1
        [np.zeros((1, 8)), a(1, T2), -a(1, 0), np.zeros((1, 8*4))], # v2
        [np.zeros((1, 8)), a(2, T2), -a(2, 0), np.zeros((1, 8*4))], # a2
        [np.zeros((1, 8)), a(3, T2), -a(3, 0), np.zeros((1, 8*4))], # j2
        [np.zeros((1, 8)), a(4, T2), -a(4, 0), np.zeros((1, 8*4))], # s2
        [np.zeros((1, 8*2)), a(1, T3), -a(1, 0), np.zeros((1, 8*3))], # v3
        [np.zeros((1, 8*2)), a(2, T3), -a(2, 0), np.zeros((1, 8*3))], # a3
        [np.zeros((1, 8*2)), a(3, T3), -a(3, 0), np.zeros((1, 8*3))], # j3
        [np.zeros((1, 8*2)), a(4, T3), -a(4, 0), np.zeros((1, 8*3))], # s3
        [np.zeros((1, 8*3)), a(1, T4), -a(1, 0), np.zeros((1, 8*2))], # v4
        [np.zeros((1, 8*3)), a(2, T4), -a(2, 0), np.zeros((1, 8*2))], # a4
        [np.zeros((1, 8*3)), a(3, T4), -a(3, 0), np.zeros((1, 8*2))], # j4
        [np.zeros((1, 8*3)), a(4, T4), -a(4, 0), np.zeros((1, 8*2))], # s4
        [np.zeros((1, 8*4)), a(1, T5), -a(1, 0), np.zeros((1, 8*1))], # v5
        [np.zeros((1, 8*4)), a(2, T5), -a(2, 0), np.zeros((1, 8*1))], # a5
        [np.zeros((1, 8*4)), a(3, T5), -a(3, 0), np.zeros((1, 8*1))], # j5
        [np.zeros((1, 8*4)), a(4, T5), -a(4, 0), np.zeros((1, 8*1))], # s5
        [np.zeros((1, 8*5)), a(1, T6), -a(1, 0)], # v6
        [np.zeros((1, 8*5)), a(2, T6), -a(2, 0)], # a6
        [np.zeros((1, 8*5)), a(3, T6), -a(3, 0)], # j6
        [np.zeros((1, 8*5)), a(4, T6), -a(4, 0)], # s6
        [a(1,0), np.zeros((1, 8*6))], # v0
        [a(2,0), np.zeros((1, 8*6))], # a0
        [a(1,T1), np.zeros((1, 8*6))], # v1
        [a(2,T1), np.zeros((1, 8*6))], # a1
        [np.zeros((1, 8)), a(1,T2), np.zeros((1, 8*5))], # v2
        [np.zeros((1, 8)), a(2,T2), np.zeros((1, 8*5))], # a2
        [np.zeros((1, 8*2)), a(1,T3), np.zeros((1, 8*4))], # v3
        [np.zeros((1, 8*2)), a(2,T3), np.zeros((1, 8*4))], # a3
        [np.zeros((1, 8*3)), a(1,T4), np.zeros((1, 8*3))], # v4
        [np.zeros((1, 8*3)), a(2,T4), np.zeros((1, 8*3))], # a4
        [np.zeros((1, 8*4)), a(1,T5), np.zeros((1, 8*2))], # v5
        [np.zeros((1, 8*4)), a(2,T5), np.zeros((1, 8*2))], # a5
        [np.zeros((1, 8*5)), a(1,T6), np.zeros((1, 8*1))], # v6
        [np.zeros((1, 8*5)), a(2,T6), np.zeros((1, 8*1))], # a6
        [np.zeros((1, 8*6)), a(1,T7)], # v7
        [np.zeros((1, 8*6)), a(2,T7)], # a7
    ])

    # Compute the inverse of A
    A_inv = np.linalg.inv(A)

    R = A_inv.T @ Q_total @ A_inv

    n_f = df.shape[0]
    n_p = 8 * 7 - n_f

    R_FF = R[:n_f, :n_f]
    R_FP = R[:n_f, n_f:n_f+n_p]
    R_PF = R[n_f:n_f+n_p, :n_f]
    R_PP = R[n_f:n_f+n_p, n_f:n_f+n_p]

    dp_star = -np.linalg.inv(R_PP) @ R_FP.T @ df

    d = np.vstack([df, dp_star])
    p = A_inv @ d

    if save:
        # Extract coefficients for each polynomial segment (x, y, z coordinates)
        p_coeffs = [p[i*8:(i+1)*8, :] for i in range(7)]
        import os
        if not os.path.exists('data'):
            os.makedirs('data')
        import json
        coefficients = {}
        for i, coeff in enumerate(p_coeffs):
            coefficients[f"p{i+1}_coeffs"] = {
                "x": coeff[:, 0].tolist(),
                "y": coeff[:, 1].tolist(),
                "z": coeff[:, 2].tolist()
            }
        with open('data/polynomial_coefficients_min_snap.json', 'w') as f:
            json.dump(coefficients, f, indent=2)

    return p

def calc_coefficients(waypoints, times):
    """
    Calculate polynomial coefficients dynamically for a minimum snap trajectory.
    """
    s = 8  # number of polynomial coefficients

    # Ensure waypoints is a numpy array
    waypoints = np.array(waypoints)

    n = len(times)  # Number of segments
    # times = calc_time(waypoints)

    Q = calc_Q(times)
    
    # construct an empty block matrix for the constraints
    df = np.zeros((5*n+3, 3)) # 

    for i in range(n):
        df[2*i, :] = waypoints[i]  # Start point of segment i
        df[2*i+1, :] = waypoints[i+1]  # End point of segment i

    A = calc_A(times)

    # Compute the inverse of A
    A_inv = np.linalg.inv(A)

    R = A_inv.T @ Q @ A_inv

    # Partition R based on the sizes of df and dp
    n_f = df.shape[0]  # number of fixed variables
    n_p = s * n - n_f  # number of free variables

    R_FF = R[:n_f, :n_f]
    R_FP = R[:n_f, n_f:n_f+n_p]
    R_PF = R[n_f:n_f+n_p, :n_f]
    R_PP = R[n_f:n_f+n_p, n_f:n_f+n_p]

    dp_star = -np.linalg.inv(R_PP) @ R_FP.T @ df

    # Assemble the complete solution vector d from df and dp_star
    d = np.vstack([df, dp_star])

    p = A_inv @ d

    return p

def calc_cost(waypoints, times, kT):
    p = calc_coefficients(waypoints, times)
    # Calculate the cost J
    J = 0.0
    for i in range(len(times)):
        T = times[i]
        Q_i = Q_snap(T)
        p_i = p[i*8:(i+1)*8, :]
        J += (p_i[:, 0].T @ Q_i @ p_i[:, 0] + 
              p_i[:, 1].T @ Q_i @ p_i[:, 1] + 
              p_i[:, 2].T @ Q_i @ p_i[:, 2])
    J += kT * np.sum(times)  # Add time penalty
    return p, J

def min_snap_gd(waypoints, times, kT, alpha=1e-10, ITER = 1000, TOL=1e-3):
    n = len(times)
    p_old, J_old = calc_cost(waypoints, times, kT)
    print(f"Initial cost: {J_old:.6f}, Initial times: {times}")
    for i in range(ITER):
        for j in range(n):
            T = times[j]
            gradQ = grad_Q(T)
            p_j = p_old[j*8:(j+1)*8, :]
            grad_J = (p_j[:, 0].T @ gradQ @ p_j[:, 0] + 
                      p_j[:, 1].T @ gradQ @ p_j[:, 1] + 
                      p_j[:, 2].T @ gradQ @ p_j[:, 2])
            # print(f"Gradient for segment {j+1}: {grad_J:.6f}")
            # print(alpha * (grad_J + kT))
            times[j] -= alpha * (grad_J + kT)
            # print(f"Updated time for segment {j+1}: {times[j]:.6f}")
            times[j] = max(0.1, times[j])  # Ensure non-negative

        p_new, J_new = calc_cost(waypoints, times, kT)
        print(f"Iteration {i}: Cost={J_new:.6f}, Times={times}")
        # if i % 100 == 0 or i == ITER - 1:
        #     print(f"Iteration {i}: Cost={J_new:.6f}, Times={times}")

        # Check for convergence
        if np.abs(J_new - J_old) < TOL:
            print(f"Converged after {i+1} iterations")
            break

        p_old, J_old = p_new, J_new


    return p_new, times

def get_coeffs_gcopter(waypoints, times):
    b = create_b(waypoints)
    M = create_M(times)

    c = solve_c(M, b)

    return c, M

def gd_gcopter(waypoints, kT, alpha=1e-10, ITER=1000, TOL=1e-3):
    times = calc_time(waypoints)
    times_old = times.copy()
    # times = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Initial guess for times
    # print(f"Initial times: {times}")
    # print(times_old)
    n = len(times)
    c,  M = get_coeffs_gcopter(waypoints, times)
    # save M.T
    np.savetxt("data/M.csv", M.T, delimiter=",")
    dKdc = calc_dKdc_2(c, times)
    G = calc_G(M, dKdc)
    s = 4
    
    # Perform gradient descent to optimize the times
    for i in range(ITER):
        for j in range(0, n-1):
            coeffs = c[j*(2*s):(j+1)*(2*s), :]
            T = times[j]
            Gj = G[s*(2*j+1):s*(2*j+3), :]
            dEdT = create_dEdT(T)
            times[j] -= alpha * calc_dWdT(coeffs, T, kT, Gj, dEdT)

        j = n-1
        coeffs = c[j*(2*s):(j+1)*(2*s), :]
        T = times[j]
        Gj = G[-s:, :]
        dEmdT = create_dEmdT(T)
        times[j] -= alpha * calc_dWdT(coeffs, T, kT, Gj, dEmdT)

        times = np.maximum(0.1, times)  # Ensure non-negative times
        DISPLAY = False
        if DISPLAY == True:
            # skip every 20 iteration
            if i % 20 == 0 or i == ITER - 1:
                print(f"Iteration {i}: Times={times}")
                # print absolute difference between times and times_old
                print(f"Times change: {np.linalg.norm(np.abs(times - times_old)):.6f}")
                p, cost = calc_cost(waypoints, times, kT) 
                print(f"Cost={cost:.6f}")
        if np.linalg.norm(np.abs(times - times_old)) < TOL:
            # print(f"Converged after {i+1} iterations")
            break
        times_old = times.copy()
        
        # p, j = calc_cost(waypoints, times, kT)
        # print(f"Cost after iteration {i}: {j:.6f}")
        c,  M = get_coeffs_gcopter(waypoints, times)
        dKdc = calc_dKdc_2(c, times)
        G = calc_G(M, dKdc)

    return c, times

def save_coeffs(c):
    # save coefficients to a file
    np.savetxt("data/coeffs.txt", c)

def save_time(times):
    # save times to a file
    with open("data/times.txt", "w") as f:
        for t in times:
            f.write(f"{t}\n")

if __name__ == "__main__":
    N = 10  # Number of repetitions for averaging
    times_list = []

    for _ in range(N):
        start = time.time()
        c, times = gd_gcopter(waypoint_list, 100, alpha=1e-4, ITER=1000, TOL=1e-2)
        end = time.time()
        times_list.append(end - start)

    avg_time = np.mean(times_list)
    print(f"Average gd_gcopter execution time over {N} runs: {avg_time:.6f} seconds")

    save_coeffs(c)
    save_time(times)