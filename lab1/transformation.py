import numpy as np
import math as mt
from graphics import Polygon, Point

class Transformation:
    def __init__(self, shape, initial_coords, dx, dy, scale_factor, angle_deg):
        self.shape = shape
        self.coords = np.array(initial_coords)
        self.dx = dx
        self.dy = dy
        self.scale_factor = scale_factor
        self.min_scale = 1 / scale_factor
        self.max_scale = scale_factor
        self.current_scale = 1
        self.scaling_up = True
        self.angle_deg = angle_deg

    def move(self, window):
        translated_coords = self._apply_translation()
        scaled_coords = self._apply_scaling(translated_coords)
        rotated_coords = self._apply_rotation(scaled_coords)

        if not self._is_within_bounds(rotated_coords, window):
            self.dx = -self.dx
            self.dy = -self.dy
            translated_coords = self._apply_translation()
            scaled_coords = self._apply_scaling(translated_coords)
            rotated_coords = self._apply_rotation(scaled_coords)

        self._update_shape(rotated_coords, window)
        self._update_scaling()

    def _apply_translation(self):
        translation_matrix = np.array([
            [1, 0, self.dx],
            [0, 1, self.dy],
            [0, 0, 1]
        ])

        return np.dot(self.coords, translation_matrix.T)

    def _apply_rotation(self, coords):
        theta_rad = mt.radians(self.angle_deg)

        center = np.mean(coords[:, :2], axis=0)

        rotation_matrix = np.array([
            [mt.cos(theta_rad), -mt.sin(theta_rad), 0],
            [mt.sin(theta_rad),  mt.cos(theta_rad), 0],
            [0, 0, 1]
        ])

        translation_to_origin = np.array([
            [1, 0, -center[0]],
            [0, 1, -center[1]],
            [0, 0, 1]
        ])

        translation_back = np.array([
            [1, 0, center[0]],
            [0, 1, center[1]],
            [0, 0, 1]
        ])

        rotated_coords = np.dot(np.dot(np.dot(coords, translation_to_origin.T), rotation_matrix.T), translation_back.T)

        return rotated_coords

    def _apply_scaling(self, coords):
        center = np.mean(coords[:, :2], axis=0)

        scaling_matrix = np.array([
            [self.current_scale, 0, 0],
            [0, self.current_scale, 0],
            [0, 0, 1]
        ])

        translation_to_origin = np.array([
            [1, 0, -center[0]],
            [0, 1, -center[1]],
            [0, 0, 1]
        ])

        translation_back = np.array([
            [1, 0, center[0]],
            [0, 1, center[1]],
            [0, 0, 1]
        ])

        scaled_coords = np.dot(np.dot(np.dot(coords, translation_to_origin.T), scaling_matrix.T), translation_back.T)

        return scaled_coords

    def _update_scaling(self):
        if self.scaling_up and self.current_scale >= self.max_scale:
            self.scaling_up = False
        elif not self.scaling_up and self.current_scale <= self.min_scale:
            self.scaling_up = True

        self.current_scale += 0.01 if self.scaling_up else -0.01

    def _is_within_bounds(self, coords, window):
        x_min, y_min = np.min(coords[:, :2], axis=0)
        x_max, y_max = np.max(coords[:, :2], axis=0)

        if x_min < 0 or x_max > window.width or y_min < 0 or y_max > window.height:
            return False
        return True

    def _update_shape(self, new_coords, window):
        self.shape.undraw()

        new_points = [Point(new_coords[i, 0], new_coords[i, 1]) for i in range(len(new_coords))]
        self.shape = Polygon(*new_points)
        self.shape.draw(window.win)

        self.coords = new_coords