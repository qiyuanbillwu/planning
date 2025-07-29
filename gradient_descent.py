from webbrowser import get
from utils import a5, Q_jerk
import constants
import numpy as np
import matplotlib.pyplot as plt
import json
import time

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


def get_coefficients(Ts):

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

# Main execution
if __name__ == "__main__":
    # Initial guess for Ts
    waypoints = [r0, r1, r2, r3, r4, r5, r6, r7]

    # set average and maximum velocity/acceleration
    v_max = 5.0
    v_avg = 3.0

    # set gradient descent parameters
    kT = 100
    alpha = 10**(-4)
    ITER = 1000
    TOL = 10**(-3)

    # start time
    start_time = time.time()

    # calculate distances and times for each segment based on average velocity
    distances = [np.linalg.norm(waypoints[i+1] - waypoints[i]) for i in range(len(waypoints)-1)]
    Ts_initial = [d / v_avg for d in distances]

    # total_time = 20
    # # Load optimized time parameters from file
    # with open('data/optimized_time_parameters.json', 'r') as f:
    #     time_data = json.load(f)
    # Ts_initial = np.array(time_data['time_proportions']) * total_time
    
    # Option to skip gradient descent
    use_gradient_descent = True  # Set to False to skip optimization
    
    if use_gradient_descent:
        # Run optimization
        results = optimize_times_gradient_descent(Ts_initial, kT, alpha, max_iterations=ITER, tolerance=TOL)
        
        # Save optimized coefficients and time parameters
        # print("\nSaving optimized coefficients and time parameters...")
        coefficients = get_coefficients(results['Ts_optimized'])
        save_optimization_results(results, coefficients)

        # Plot optimization history
        plot_optimization_history(results)
    else:
        # Use initial time parameters directly
        print("Skipping gradient descent optimization...")
        print(f"Using initial time parameters: {Ts_initial}")
        
        # Calculate cost for initial parameters
        initial_cost = J(Ts_initial, kT)
        total_time = np.sum(Ts_initial)
        time_proportions = np.array(Ts_initial) / total_time
        
        # Create results dictionary for consistency
        results = {
            'Ts_optimized': np.array(Ts_initial),
            'final_cost': initial_cost,
            'total_time': total_time,
            'time_proportions': time_proportions,
            'Ts_history': np.array([Ts_initial]),
            'cost_history': [initial_cost],
            'optimization_time_seconds': 0.0,
            'iterations': 0,
            'converged': True
        }
        
        # Save coefficients and time parameters
        print("\nSaving coefficients and time parameters...")
        coefficients = get_coefficients(Ts_initial)
        save_optimization_results(results, coefficients)
        
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Initial cost: {initial_cost:.6f}")
        print("Time proportions:")
        for i, (T, prop) in enumerate(zip(Ts_initial, time_proportions)):
            print(f"  Segment {i+1}: {T:.2f}s ({prop:.1%})")