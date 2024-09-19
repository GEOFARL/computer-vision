from graphics import Point, Rectangle
import numpy as np


class Transformation:
    def __init__(self, shape, dx, dy, scale_factor, min_size, max_size):
        self.shape = shape
        self.dx = dx
        self.dy = dy
        self.scale_factor = scale_factor
        self.min_size = min_size
        self.max_size = max_size

    def translate(self, window):
        new_coordinates = self._calculate_new_coordinates()
        self._check_and_reverse_direction(new_coordinates, window)
        self.shape.move(self.dx, self.dy)

    def _calculate_new_coordinates(self):
        current_coords = np.array([[self.shape.p1.x, self.shape.p1.y], 
                                   [self.shape.p2.x, self.shape.p2.y]])
        delta = np.array([self.dx, self.dy])
        new_coords = current_coords + delta
        return new_coords

    def _check_and_reverse_direction(self, new_coords, window):
        new_x1, new_y1 = new_coords[0]
        new_x2, new_y2 = new_coords[1]
        if new_x1 < 0 or new_x2 > window.width:
            self.dx = -self.dx
        if new_y1 < 0 or new_y2 > window.height:
            self.dy = -self.dy

    def scale(self, scaling_up, window):
        factor, new_width, new_height = self._calculate_scaling_factor(scaling_up)
        new_dimensions = self._calculate_new_dimensions(new_width, new_height)
        
        if self._is_within_size_limits(new_width, new_height) and self._is_within_bounds(new_dimensions, window):
            self._update_shape(new_dimensions, window)

    def _calculate_scaling_factor(self, scaling_up):
        factor = self.scale_factor if scaling_up else 1 / self.scale_factor
        current_width = self.shape.p2.x - self.shape.p1.x
        current_height = self.shape.p2.y - self.shape.p1.y
        new_width = current_width * factor
        new_height = current_height * factor
        return factor, new_width, new_height

    def _calculate_new_dimensions(self, new_width, new_height):
        center = self.shape.getCenter()
        new_coords = np.array([center.x, center.y]) + np.array([[-new_width / 2, -new_height / 2], 
                                                                [new_width / 2, new_height / 2]])
        return new_coords

    def _is_within_size_limits(self, new_width, new_height):
        return self.min_size <= new_width <= self.max_size and self.min_size <= new_height <= self.max_size

    def _is_within_bounds(self, new_coords, window):
        new_x1, new_y1 = new_coords[0]
        new_x2, new_y2 = new_coords[1]
        return 0 <= new_x1 <= window.width and 0 <= new_x2 <= window.width and \
               0 <= new_y1 <= window.height and 0 <= new_y2 <= window.height

    def _update_shape(self, new_coords, window):
        self.shape.undraw()
        new_x1, new_y1 = new_coords[0]
        new_x2, new_y2 = new_coords[1]
        self.shape = Rectangle(Point(new_x1, new_y1), Point(new_x2, new_y2))
        self.shape.draw(window.win)
