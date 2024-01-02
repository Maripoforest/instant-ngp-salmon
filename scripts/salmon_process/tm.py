import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
# Create a sample 4x4 transformation matrix
# transform_matrix = np.array([
#     [0.866, -0.5, 0.0, 2.0],
#     [0.5, 0.866, 0.0, 3.0],
#     [0.0, 0.0, 1.0, 1.0],
#     [0.0, 0.0, 0.0, 1.0]
# ])

with open("./transforms.json", 'r') as json_file:
    data_dict = json.load(json_file)
with open("./original.json", 'r') as json_file:
    original_dict = json.load(json_file)

matrix_data = data_dict["cams"]
# ================================================
# matrix_data = list()
# _matrix_data = original_dict["frames"]
# for i in range(len(_matrix_data)):
#     matrix_data.append(_matrix_data[i]["transform_matrix"])
# print(matrix_data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
i = 0
for transform_matrix in matrix_data:
    if i == 10:
        break
    else:
        i += 1

    # theta = np.radians(90)
    # rotation_matrix = np.array([
    #     [np.cos(theta), -np.sin(theta), 0, 0],
    #     [np.sin(theta), np.cos(theta), 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]
    # ])

    transform_matrix = np.array(transform_matrix)
    
    theta = np.radians(90)
    # rotation_matrix = np.array([
    #     [np.cos(theta), -np.sin(theta), 0, 0],
    #     [np.sin(theta), np.cos(theta), 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]
    # ])
    # transform_matrix = np.dot(rotation_matrix, transform_matrix)
    
    rotation_matrix = transform_matrix[:3, :3]
    _rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    rotation_matrix = np.dot(rotation_matrix, _rotation_matrix)

    translation_vector = transform_matrix[:3, 3]

    # Translate the orientation axes to the position
    x_axis = rotation_matrix[:, 0] + translation_vector
    y_axis = rotation_matrix[:, 1] + translation_vector
    z_axis = rotation_matrix[:, 2] + translation_vector

    # Plot axes
    ax.quiver(*translation_vector, *x_axis-translation_vector, color='r', label='X-Axis')
    ax.quiver(*translation_vector, *y_axis-translation_vector, color='g', label='Y-Axis')
    ax.quiver(*translation_vector, *z_axis-translation_vector, color='b', label='Z-Axis')

    # Plot position as a point
    ax.scatter(*translation_vector, color='k', marker='o', s=100, label='Position')

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set limits for the plot
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_zlim([-6, 6])

# Display the plot
plt.title('Transformation Matrix Visualization')
plt.show()
