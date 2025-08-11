

# Extract 8 waypoints from waypoints.txt
import numpy as np

r0 = np.array([1, 1, 0.5])
r1 = np.array([0, 1.5, 1])
r2 = np.array([-0.8, 1, 2])
r3 = np.array([-1.5, 0, 2.5])
r4 = np.array([-1, -1, 1.5])
r5 = np.array([0.25, -0.75, 1])
r6 = np.array([1, 0, 1.5])
r7 = np.array([0, 0, 2])
waypoint_list = np.array([r0, r1, r2, r3, r4, r5, r6, r7])