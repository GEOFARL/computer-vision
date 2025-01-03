from graphics import Polygon, Point, Line
from graphics_window import GraphicsWindow
from transformation import Transformation
import time

def setup_graphics(width, height):
    window = GraphicsWindow("2D Transformations", width, height)

    initial_coords = [
        [100, 450, 1],
        [200, 450, 1],
        [200, 500, 1],
        [100, 500, 1]
    ]

    polygon = Polygon(Point(100, 450), Point(200, 450), Point(200, 500), Point(100, 500))
    polygon.setFill('red')
    polygon.setOutline('red')
    polygon.draw(window.win)
    
    return window, polygon, initial_coords

def run_animation(window, transformation, num_frames):
    prev_center = transformation.get_center()

    for _ in range(num_frames):
        if window.check_mouse():
            return
        time.sleep(0.05)
        
        transformation.move(window)

        new_center = transformation.get_center()
        line = Line(Point(prev_center[0], prev_center[1]), Point(new_center[0], new_center[1]))
        line.setOutline('blue')
        line.draw(window.win)
        prev_center = new_center

def main():
    width, height = 600, 600
    window, polygon, initial_coords = setup_graphics(width, height)
    
    transformation = Transformation(
        shape=polygon,
        initial_coords=initial_coords,
        dx=10, dy=-10,
        scale_factor=1.1,
        angle_deg=10
    )

    num_frames = 300
    run_animation(window, transformation, num_frames)
    window.check_mouse()
    window.close()


if __name__ == "__main__":
    main()
