import cv2
import numpy as np
import logging
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)

def display_image(title, image, cmap=None):
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def merge_contours(contours, proximity_threshold=30):
    merged_contours = []
    for cnt in contours:
        merged = False
        for idx, merged_cnt in enumerate(merged_contours):
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroid = (cx, cy)
                dist = cv2.pointPolygonTest(merged_cnt, centroid, True)
                if abs(dist) < proximity_threshold:
                    merged_contours[idx] = np.vstack((merged_cnt, cnt))
                    merged = True
                    break
        if not merged:
            merged_contours.append(cnt)
    return merged_contours

image = cv2.imread('./lab4/image.jpeg')
scale_percent = 30
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image_resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

brightness_adjust = 0
contrast_adjust = 1
adjusted_image = cv2.convertScaleAbs(image_resized, alpha=contrast_adjust, beta=brightness_adjust)

gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray_image, (7, 7), 0)
display_image('Blurred Image', blurred, cmap='gray')

edges = cv2.Canny(blurred, threshold1=120, threshold2=250)
display_image('Edges', edges, cmap='gray')

kernel = np.ones((5, 5), np.uint8)
morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
display_image('Morphological Closing', morph, cmap='gray')

contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
merged_contours = merge_contours(contours, proximity_threshold=50)

min_area, max_area = 2250, 1000000
output_image = adjusted_image.copy()

for contour in merged_contours:
    area = cv2.contourArea(contour)
    if min_area < area < max_area:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.3 < aspect_ratio < 0.75:
            logging.info(f"Detected pedestrian-like contour. BBox: ({x}, {y}, {w}, {h})")
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

display_image('Detected Pedestrians', cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
