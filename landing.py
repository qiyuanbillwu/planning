# landing on a translating and oscillating platform

import numpy as np
from constants import v_avg

r0 = np.array([-100, 0, 20])
v0 = np.array([0, 0, 0])
a0 = np.array([0, 0, 0])
j0 = np.array([0, 0, 0])

def r(t):
    return np.array([t, np.sin(t), 0])
def v(t):
    return np.array([1, np.cos(t), 0])
def a(t):
    return np.array([0, -np.sin(t), 0])
def j(t):
    return np.array([0, -np.cos(t), 0])

# Calculate the time it takes to reach the target position
v_avg = 10
t_guess = (200 + np.sqrt(40000 + 41600*(v_avg**2+1))) / (2 * (v_avg**2 + 1))
print("Initial guess for t:", t_guess)

# calculate final waypoint
rf = r(t_guess)
vf = v(t_guess)
af = a(t_guess)
jf = j(t_guess)

# generate intermediate waypoints