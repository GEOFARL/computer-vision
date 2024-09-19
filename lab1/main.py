from graphics import Rectangle, Point, update
from graphics_window import GraphicsWindow
from transformation import Transformation
import time
import random

def setup_graphics(width, height):
    window = GraphicsWindow("2D Transformations", width, height)
    rect = Rectangle(Point(100, 450), Point(200, 500))
    rect.draw(window.win)
    return window, rect

def run_animation(window, transformation, num_frames):
    for i in range(num_frames):
        if window.check_mouse():
            return
        time.sleep(0.05)
        transformation.translate(window)
        if i % 20 == 0:
            perform_scaling(transformation, window)

def perform_scaling(transformation, window):
    scaling_up = random.choice([True, False])
    for _ in range(random.randint(3, 5)):
        transformation.scale(scaling_up, window)
        time.sleep(0.1)
        update(30)

def main():
    width, height = 600, 600
    window, rect = setup_graphics(width, height)
    transformation = Transformation(shape=rect, dx=10, dy=-10, scale_factor=1.2, min_size=50, max_size=min(width, height) * 0.5)
    num_frames = 300
    run_animation(window, transformation, num_frames)
    window.check_mouse()
    window.close()

if __name__ == "__main__":
    main()
