import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

def define_cube():
    return np.array([
        [-1, -1, -1, 1], [1, -1, -1, 1], [1, 1, -1, 1], [-1, 1, -1, 1],
        [-1, -1, 1, 1], [1, -1, 1, 1], [1, 1, 1, 1], [-1, 1, 1, 1]
    ])

def axonometric_projection(shape, phi_deg=35.264, theta_deg=45):
    phi = np.radians(phi_deg)
    theta = np.radians(theta_deg)
    projection_matrix = np.array([
        [np.cos(theta), np.sin(theta) * np.sin(phi), 0, 0],
        [0, np.cos(phi), 0, 0],
        [np.sin(theta), -np.cos(theta) * np.sin(phi), 0, 0],
        [0, 0, 0, 1]
    ])
    return shape @ projection_matrix

def rotate(shape, angle_deg, axis):
    theta = np.radians(angle_deg)
    if axis == 'y':
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1]
        ])
    return shape @ rotation_matrix

def init_plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig, ax

def update(frame, verts, ax, fig):
    ax.clear()
    rotated = rotate(verts, frame * 2, 'y')
    projected = axonometric_projection(rotated) 
    faces = [
        [projected[0, :3], projected[1, :3], projected[2, :3], projected[3, :3]],
        [projected[4, :3], projected[5, :3], projected[6, :3], projected[7, :3]],
        [projected[0, :3], projected[3, :3], projected[7, :3], projected[4, :3]],
        [projected[2, :3], projected[1, :3], projected[5, :3], projected[6, :3]],
        [projected[0, :3], projected[1, :3], projected[5, :3], projected[4, :3]],
        [projected[3, :3], projected[2, :3], projected[6, :3], projected[7, :3]]
    ]
    collection = Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25)
    ax.add_collection3d(collection)
    return collection,

fig, ax = init_plot()
cube = define_cube()

ani = FuncAnimation(fig, update, frames=np.arange(0, 90, 1), fargs=(cube, ax, fig), repeat=True)

plt.show()
