import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from trajectory_calcs import compute_full_state
from constants import T1, T2

# Gate position (from min_jerk_x, min_jerk_y, min_jerk_z)
x1 = 0   # from min_jerk_x: p1 = 0
y1 = -1  # from min_jerk_y: p1 = -1
z1 = 2   # from min_jerk_z: p1 = 2
# Rectangle orientation (theta): vertical, so theta = 0
# If you want to compute theta from trajectory, you can do so, but here we set it to 0
theta = 0

# Time setup
dt = 0.01
t0 = 0
t2 = T1 + T2

ts = np.arange(t0, t2, dt)
# Get trajectory states
states = [compute_full_state(t) for t in ts]
pos = np.array([s['r'] for s in states])
adhat = np.array([s['a'] + np.array([0,0,9.81]) for s in states])
adhat = adhat / np.linalg.norm(adhat, axis=1)[:,None]

# Setup figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=15, azim=160)  # type: ignore

# Rectangle parameters (x-axis aligned)
rect_center = np.array([x1, y1, z1])
width, height = 0.4, 0.7
rotation_angle = theta * 180 / np.pi  # degrees

def create_rotated_rectangle(center, width, height, angle):
    corners = np.array([
        [0, -width/2, -height/2],
        [0, width/2, -height/2],
        [0, width/2, height/2],
        [0, -width/2, height/2]
    ])
    theta = np.radians(angle)
    rot = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    return np.dot(corners, rot.T) + center

# Create and add rectangle
rect_verts = create_rotated_rectangle(rect_center, width, height, rotation_angle)
rect = Poly3DCollection([rect_verts], alpha=0.3, facecolor='green', edgecolor='k')
ax.add_collection3d(rect)  # type: ignore

# Animation elements
line, = ax.plot([], [], [], 'b', label='Trajectory')
point, = ax.plot([], [], [], 'ko', markersize=8)
quiver = None
# Use fig.text for 2D overlay text (always works)
time_text = fig.text(0.02, 0.95, '')

# Axis limits and labels
ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(0, 4),
       xlabel='x', ylabel='y', zlabel='z')
ax.legend()

def init():
    line.set_data([], [])
    line.set_3d_properties([])  # type: ignore
    point.set_data([], [])
    point.set_3d_properties([])  # type: ignore
    global quiver
    if quiver is not None:
        quiver.remove()
        quiver = None
    time_text.set_text('')
    return line, point, time_text

def update(frame):
    if frame > 0:
        line.set_data(pos[:frame, 0], pos[:frame, 1])
        line.set_3d_properties(pos[:frame, 2])  # type: ignore
        p = pos[frame-1]
        point.set_data([p[0]], [p[1]])
        point.set_3d_properties([p[2]])  # type: ignore
        global quiver
        if quiver is not None:
            quiver.remove()
        quiver_vec = adhat[frame-1]
        quiver = ax.quiver(p[0], p[1], p[2], quiver_vec[0], quiver_vec[1], quiver_vec[2], length=0.5, color='r')
        time_text.set_text(f'Time: {ts[frame-1]:.2f}s')
    return line, point, time_text

fps = 40
ani = FuncAnimation(fig, update, frames=len(ts), init_func=init,
                   interval=1000/fps, blit=False)

plt.show()
