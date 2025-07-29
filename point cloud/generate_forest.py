from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D  # This import is needed for 3D plotting

def generate_tilted_cylinders(num_cylinders):
    resolution = 0.1
    min_dim = [-2, -2, 0]
    max_dim = [2, 2, 4]  # 4x4x4 box, z from 0 to 4

    cylinder_points = []
    cylinder_centers = []
    cylinder_radius = []

    while len(cylinder_radius) < num_cylinders:
        valid = True
        radius = np.random.uniform(0.1, 0.25)
        length = np.random.uniform(1, 3)
        cx = np.random.uniform(-2 + length/2, 2 - length/2)
        cy = np.random.uniform(-2 + radius, 2 - radius)
        cz = np.random.uniform(radius, 4 - radius)

        for i in range(len(cylinder_centers)):
            cx_ref, cy_ref, cz_ref, length_ref, radius_ref = cylinder_centers[i]
            dist_centers = np.linalg.norm([cx - cx_ref, cy - cy_ref, cz - cz_ref])
            dist_radius = radius + radius_ref
            if (dist_centers <= dist_radius):
                valid = False
                break
        if valid is False:
            continue

        # Create a cylinder along the x-axis
        thetas = np.arange(0, 2 * np.pi, resolution / radius)
        x_vals = np.arange(-length/2, length/2, resolution)
        points = []
        for theta in thetas:
            y = radius * np.cos(theta)
            z = radius * np.sin(theta)
            for x in x_vals:
                points.append(np.array([x, y, z]))
        points = np.array(points)

        # Generate random rotation
        alpha = np.random.uniform(0, 2 * np.pi) # Z-axis rotation
        beta = np.random.uniform(0, np.pi)     # Y-axis rotation
        gamma = np.random.uniform(0, 2 * np.pi) # X-axis rotation

        Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                       [np.sin(alpha), np.cos(alpha),  0],
                       [0,             0,              1]])
        Ry = np.array([[np.cos(beta),  0, np.sin(beta)],
                       [0,             1, 0],
                       [-np.sin(beta), 0, np.cos(beta)]])
        Rx = np.array([[1, 0,            0],
                       [0, np.cos(gamma), -np.sin(gamma)],
                       [0, np.sin(gamma), np.cos(gamma)]])
        R = Rz @ Ry @ Rx

        rotated_points = points @ R.T
        translated_points = rotated_points + np.array([cx, cy, cz])

        # Check if all points are within the box
        if (
            np.all(translated_points[:, 0] >= min_dim[0]) and np.all(translated_points[:, 0] <= max_dim[0]) and
            np.all(translated_points[:, 1] >= min_dim[1]) and np.all(translated_points[:, 1] <= max_dim[1]) and
            np.all(translated_points[:, 2] >= min_dim[2]) and np.all(translated_points[:, 2] <= max_dim[2])
        ):
            cylinder_centers.append([cx, cy, cz, length, radius])
            cylinder_radius.append(radius)
            cylinder_points.extend(translated_points)
        # else: try again

    forest_points = cylinder_points

    # add floor
    for x in np.arange(-2, 2, resolution):
        for y in np.arange(-2, 2, resolution):
            forest_points.append(np.array([x, y, 0.]))

    # add walls
    for x in np.arange(-2, 2, resolution):
        for z in np.arange(0, 4, resolution):
            forest_points.append(np.array([x, -2, z]))
            forest_points.append(np.array([x, 2, z]))

    for y in np.arange(-2, 2, resolution):
        for z in np.arange(0, 4, resolution):
            forest_points.append(np.array([-2, y, z]))
            forest_points.append(np.array([2, y, z]))

    # add ceiling
    for x in np.arange(-2, 2, resolution):
        for y in np.arange(-2, 2, resolution):
            forest_points.append(np.array([x, y, 4.]))

    point_cloud_np = np.array(forest_points, dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

    return pcd


# Generate the point cloud
point_cloud = generate_tilted_cylinders(10)

# Save point cloud to PCD file
output_file = 'tilted_cylinders.pcd'
o3d.io.write_point_cloud(output_file, point_cloud)

points = np.asarray(point_cloud.points)

# 3D scatter plot using Axes3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')  # Axes3D is used here
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Generated Forest Point Cloud')
ax.set_aspect('equal')
# plt.tight_layout()
plt.show()

print(f"Point cloud saved to {output_file}")