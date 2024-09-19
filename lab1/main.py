from graphics import *
import time

class GraphicsWindow:
  def __init__(self, title, width, height):
    self.title = title
    self.width = width
    self.height = height
    self.win = GraphWin(self.title, self.width, self.height)
    self.win.setBackground("white")

  def close(self):
    self.win.close()

  def get_mouse(self):
    return self.win.getMouse()
  
  def add_shape(self, shape):
    shape.draw(self.win)

class RectangleTransformation:
  def __init__(self, x1, y1, x2, y2):
    self.rect = Rectangle(Point(x1, y1), Point(x2, y2))
    self.dx = 10
    self.dy = -10

  def draw(self, window):
    window.add_shape(self.rect)

  def translate(self):
    self.rect.move(self.dx, self.dy)

  def perform_transformation(self, window, num_frames):
    for _ in range(num_frames):
      time.sleep(0.05)
      self.translate()

def main():
  width, height = 600, 600
  window = GraphicsWindow("2D Transformations", width, height)

  rect_transformation = RectangleTransformation(100, 450, 200, 500)
  rect_transformation.draw(window)

  rect_transformation.perform_transformation(window, 60)

  window.get_mouse()
  window.close()

if __name__ == "__main__":
  main()