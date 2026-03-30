import numpy as np
# Global constants for trajectory planning

# Time intervals for trajectory segments
T1 = 5  # Duration of first segment
T2 = 5  # Duration of second segment

# Total trajectory duration
T_TOTAL = T1 + T2

# Gate time (connection point between segments)
T_GATE = T1

# Physical constants
g = 9.81  # Gravitational acceleration (m/s²) 

# Vehicle constants
m = 0.745  # mass of drone [kg]
l = 0.115   # meters [m]
Cd = 0.01   # drag coefficient of propellers [PLACEHOLDER]
Cl = 0.1    # lift coefficent of propellers  [PLACEHOLDER]

# make sure this is agreement with the allocation matrix
J = np.diag([0.00225577, 0.00360365, 0.00181890]) # [kg/m2]
d = Cd / Cl

# maximum and average velocity/acceleration
v_avg = 1.0  # Average velocity (m/s)
a_avg = 3.0  # Average acceleration (m/s²)
v_max = 5.0  # Maximum velocity (m/s)
a_max = 5.0  # Maximum acceleration (m/s²)  
