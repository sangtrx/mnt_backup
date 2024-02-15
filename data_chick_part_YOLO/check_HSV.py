import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from PIL import ImageOps
import numpy as np
import cv2


def print_hsv_values(image_path, x, y):
    # Load the image
    img = Image.open(image_path)
    img_np = np.array(img)

    # Convert the image to HSV
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Print the HSV values at the specified location
    print("HSV values at", x, y, ":", hsv[int(y), int(x)])

# Call the function for the specified image
image_path = "/mnt/tqsang/chicken_part1/part1_cropped_frames/00045198_L.png"
x, y = (357.95025634765625 - 364.3116760253906/2), 362.3189697265625
print_hsv_values(image_path, x, y)

# Blueish color HSV value
h, s, v = 104, 239, 109

# Tolerance values for H, S, and V components
h_tolerance = 10
s_tolerance = 50
v_tolerance = 50

# Create lower and upper boundaries for the blueish color in the HSV color space
lower_blueish = np.array([h - h_tolerance, s - s_tolerance, v - v_tolerance])
upper_blueish = np.array([h + h_tolerance, s + s_tolerance, v + v_tolerance])

print("Lower boundary for blueish color: ", lower_blueish)
print("Upper boundary for blueish color: ", upper_blueish)