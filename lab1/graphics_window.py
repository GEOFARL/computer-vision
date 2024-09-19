from graphics import GraphWin

class GraphicsWindow:
    def __init__(self, title, width, height):
        self.title = title
        self.width = width
        self.height = height
        self.win = GraphWin(self.title, self.width, self.height)
        self.win.setBackground('white')
        self.win.autoflush = False
    
    def close(self):
        self.win.close()
    
    def check_mouse(self):
        return self.win.checkMouse()
