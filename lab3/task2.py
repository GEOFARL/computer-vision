import cv2
import numpy as np

image = cv2.imread('./lab3/image3.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_prewitt(image):
    prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    prewitt_x_edges = cv2.filter2D(image, -1, prewitt_x)
    prewitt_y_edges = cv2.filter2D(image, -1, prewitt_y)

    prewitt_combined = cv2.bitwise_or(prewitt_x_edges, prewitt_y_edges)

    return prewitt_combined

prewitt_edges = apply_prewitt(gray_image)

cv2.imshow('Prewitt Edge Detection', prewitt_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
