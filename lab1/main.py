from graphics import *
import time
import random

class GraphicsWindow:
    def __init__(self, title, width, height):
        self.title = title
        self.width = width
        self.height = height
        self.win = GraphWin(self.title, self.width, self.height)
        self.win.setBackground('white')
        self.win.autoflush = False  # Manage updates manually to control the animation timing
    
    def close(self):
        self.win.close()
    
    def get_mouse(self):
        return self.win.checkMouse()  # Non-blocking check for mouse click

class RectangleTransformation:
    def __init__(self, window, x1, y1, x2, y2):
        self.window = window
        self.rect = Rectangle(Point(x1, y1), Point(x2, y2))
        self.rect.draw(window.win)
        self.dx = 10
        self.dy = -10
        self.scale_factor = 1.2
        self.min_size = 50
        self.max_size = min(self.window.width, self.window.height) * 0.5

    def translate(self):
        """ Translates the rectangle and reverses direction on boundary collision. """
        new_x1 = self.rect.p1.x + self.dx
        new_y1 = self.rect.p1.y + self.dy
        new_x2 = self.rect.p2.x + self.dx
        new_y2 = self.rect.p2.y + self.dy

        # Check for boundary collision and reverse direction if needed
        if new_x1 < 0 or new_x2 > self.window.width:
            self.dx = -self.dx
        if new_y1 < 0 or new_y2 > self.window.height:
            self.dy = -self.dy

        self.rect.move(self.dx, self.dy)

    def scale(self, scaling_up):
        """ Scales the rectangle up or down based on the scaling_up flag and reverses direction when size limits are reached. """
        factor = self.scale_factor if scaling_up else 1 / self.scale_factor
        center = self.rect.getCenter()
        current_width = self.rect.p2.x - self.rect.p1.x
        current_height = self.rect.p2.y - self.rect.p1.y
        
        # Calculate proposed new dimensions
        new_width = current_width * factor
        new_height = current_height * factor

        # Check size constraints and reverse direction if necessary
        if new_width < self.min_size or new_width > self.max_size or new_height < self.min_size or new_height > self.max_size:
            # Reverse scaling direction by inverting the scaling factor
            factor = 1 / factor
            new_width = current_width * factor
            new_height = current_height * factor

        # Calculate new rectangle coordinates based on the new factor
        new_x1 = center.x - new_width / 2
        new_x2 = center.x + new_width / 2
        new_y1 = center.y - new_height / 2
        new_y2 = center.y + new_height / 2

        # Check window boundaries to ensure the rectangle stays within the window
        if 0 <= new_x1 and new_x2 <= self.window.width and 0 <= new_y1 and new_y2 <= self.window.height:
            self.rect.undraw()
            self.rect = Rectangle(Point(new_x1, new_y1), Point(new_x2, new_y2))
            self.rect.draw(self.window.win)

    def perform_transformations(self, num_frames):
        for i in range(num_frames):
            if self.window.get_mouse():  # Stop if mouse click detected
                break
            time.sleep(0.05)
            self.translate()
            if i % 20 == 0:  # Random scaling bursts every 20 frames
                scaling_up = random.choice([True, False])  # Randomly choose scaling direction
                for _ in range(random.randint(3, 5)):  # Perform 3 to 5 scaling operations
                    self.scale(scaling_up)
                    time.sleep(0.1)  # Short pause between scales for visual effect
                    update(30)  # Update window with animations

def main():
    width, height = 600, 600
    window = GraphicsWindow("2D Transformations with Random Scaling Bursts", width, height)

    # Initialize rectangle and perform transformations
    rect_transformation = RectangleTransformation(window, 100, 450, 200, 500)
    rect_transformation.perform_transformations(300)

    window.get_mouse()
    window.close()

if __name__ == "__main__":
    main()
