import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import lagrange
import time

def axonometric_projection(vertices):
    axonometric_matrix = np.array([[1, 0, 0],
                                   [0, 1, 0],
                                   [0.5, 0.5, 1]])
    return np.dot(vertices, axonometric_matrix.T)

def create_parallelepiped():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    return vertices

def lagrange_interpolation(start, end, num_steps):
    times = np.linspace(0, 1, 3)
    values = np.array([start, (start + end) / 2, end])
    interpolated_steps = []

    for i in range(3):
        poly = lagrange(times, values[:, i])
        interpolated_steps.append([poly(t) for t in np.linspace(0, 1, num_steps)])
    
    return np.array(interpolated_steps).T

def sort_faces_by_depth(faces):
    depths = [np.mean([vertex[2] for vertex in face]) for face in faces]
    sorted_faces = [face for _, face in sorted(zip(depths, faces), key=lambda x: x[0], reverse=True)]
    return sorted_faces

def plot_parallelepiped(ax, vertices):
    faces = [[vertices[j] for j in [0, 1, 2, 3]],
             [vertices[j] for j in [4, 5, 6, 7]],
             [vertices[j] for j in [0, 1, 5, 4]],
             [vertices[j] for j in [2, 3, 7, 6]],
             [vertices[j] for j in [0, 3, 7, 4]],
             [vertices[j] for j in [1, 2, 6, 5]]]

    sorted_faces = sort_faces_by_depth(faces)

    ax.clear()
    for face in sorted_faces:
        ax.add_collection3d(Poly3DCollection([face], facecolors='cyan', linewidths=1, edgecolors='r', alpha=.6))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.draw()
    plt.pause(0.01)

def animate_parallelepiped(vertices, num_steps=30):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    start_vertices = vertices.copy()

    translation = np.array([2, 2, 0])
    scale_factor = 3

    end_vertices = (vertices * np.array([1, 1, scale_factor])) + translation

    interpolated_steps = []
    for i in range(8):
        interpolated_steps.append(lagrange_interpolation(start_vertices[i], end_vertices[i], num_steps))
    
    interpolated_steps = np.array(interpolated_steps)

    for step in range(num_steps):
        new_vertices = interpolated_steps[:, step, :]
        vertices_projected = axonometric_projection(new_vertices)
        plot_parallelepiped(ax, vertices_projected)
        time.sleep(0.05)

    plt.close(fig)

vertices = create_parallelepiped()
animate_parallelepiped(vertices, num_steps=100)
