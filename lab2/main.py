import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

def load_image(file_name: str):
    try:
        image = Image.open(file_name)
        width, height = image.size
        pixels = np.array(image)
        return pixels, width, height
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

def save_image(pixels, file_name: str):
    image = Image.fromarray(pixels.astype('uint8'))
    image.save(file_name)

def show_image(pixels):
    plt.imshow(pixels)
    plt.show()

def gradient_from_center(width, height):
    center_x, center_y = width // 2, height // 2
    max_distance = math.sqrt(center_x**2 + center_y**2)
    mask = np.zeros((height, width))
    
    for y in range(height):
        for x in range(width):
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            mask[y, x] = distance / max_distance
    return mask

def gradient_diagonal(width, height):
    mask = np.zeros((height, width))
    max_distance = math.sqrt(width**2 + height**2)
    
    for y in range(height):
        for x in range(width):
            distance = math.sqrt(x**2 + y**2)
            mask[y, x] = distance / max_distance
    return mask

def gradient_toward_center(width, height):
    mask = gradient_from_center(width, height)
    return 1 - mask

def apply_grayscale_with_gradient(pixels, mask):
    gray_pixels = np.zeros_like(pixels)
    for y in range(pixels.shape[0]):
        for x in range(pixels.shape[1]):
            r, g, b = pixels[y, x]
            gray = int((r + g + b) / 3)
            intensity = int(gray * mask[y, x])
            gray_pixels[y, x] = [intensity, intensity, intensity]
    return gray_pixels

def apply_sepia_with_gradient(pixels, mask, depth=20):
    sepia_pixels = np.zeros_like(pixels)
    for y in range(pixels.shape[0]):
        for x in range(pixels.shape[1]):
            r, g, b = pixels[y, x]
            gray = int((r + g + b) / 3)
            r = min(255, gray + 2 * depth)
            g = min(255, gray + depth)
            b = gray
            intensity = mask[y, x]
            sepia_pixels[y, x] = [int(r * intensity), int(g * intensity), int(b * intensity)]
    return sepia_pixels

def apply_negative_with_gradient(pixels, mask):
    negative_pixels = np.zeros_like(pixels)
    for y in range(pixels.shape[0]):
        for x in range(pixels.shape[1]):
            r, g, b = pixels[y, x]
            r = int((255 - r) * mask[y, x])
            g = int((255 - g) * mask[y, x])
            b = int((255 - b) * mask[y, x])
            negative_pixels[y, x] = [r, g, b]
    return negative_pixels

def apply_brightness_with_gradient(pixels, mask, brightness_factor):
    brightness_pixels = np.zeros_like(pixels)
    for y in range(pixels.shape[0]):
        for x in range(pixels.shape[1]):
            r, g, b = pixels[y, x]
            r = min(255, int(r * (1 + brightness_factor * mask[y, x])))
            g = min(255, int(g * (1 + brightness_factor * mask[y, x])))
            b = min(255, int(b * (1 + brightness_factor * mask[y, x])))
            brightness_pixels[y, x] = [r, g, b]
    return brightness_pixels

def main():
    print("Welcome to the Image Color Correction CLI!")
    file_name = input("Enter the file name of the image you want to edit: ").strip()

    pixels, width, height = load_image(file_name)

    print("\nSelect a transformation type:")
    print("1 - Grayscale")
    print("2 - Sepia")
    print("3 - Negative")
    print("4 - Brightness")
    transform_choice = int(input("Enter the number corresponding to your choice: ").strip())

    print("\nSelect a gradient type:")
    print("1 - From the center")
    print("2 - Diagonal (top-left to bottom-right)")
    print("3 - Toward the center")
    gradient_choice = int(input("Enter the number corresponding to your gradient choice: ").strip())

    if gradient_choice == 1:
        mask = gradient_from_center(width, height)
    elif gradient_choice == 2:
        mask = gradient_diagonal(width, height)
    elif gradient_choice == 3:
        mask = gradient_toward_center(width, height)
    else:
        print("Invalid gradient choice!")
        sys.exit(1)

    if transform_choice == 1:
        print("\nApplying Grayscale transformation...")
        result_pixels = apply_grayscale_with_gradient(pixels, mask)
    elif transform_choice == 2:
        depth = int(input("Enter sepia depth (recommended range: 20-50): ").strip())
        print("\nApplying Sepia transformation...")
        result_pixels = apply_sepia_with_gradient(pixels, mask, depth)
    elif transform_choice == 3:
        print("\nApplying Negative transformation...")
        result_pixels = apply_negative_with_gradient(pixels, mask)
    elif transform_choice == 4:
        brightness_factor = float(input("Enter brightness factor (-1 to 1): ").strip())
        print("\nApplying Brightness transformation...")
        result_pixels = apply_brightness_with_gradient(pixels, mask, brightness_factor)
    else:
        print("Invalid transformation choice!")
        sys.exit(1)

    print("\nDisplaying the result...")
    show_image(result_pixels)

    output_file = input("Enter the file name to save the output image: ").strip()
    save_image(result_pixels, output_file)
    print(f"\nImage saved as {output_file}!")

if __name__ == "__main__":
    main()
