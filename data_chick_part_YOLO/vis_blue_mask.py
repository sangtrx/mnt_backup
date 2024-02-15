import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from PIL import ImageOps
import numpy as np
import cv2

def visualize_blue_mask(input_image_path):
    img = Image.open(input_image_path)
    img_np = np.array(img)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Define the range for blue color in HSV
    lower_blue = np.array([100, 60, 60])
    upper_blue = np.array([140, 255, 255])

    # Threshold the HSV image to get only blue colors
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Save the blue mask to a file
    Image.fromarray(blue_mask).save("blue_mask.png")


# input_image_path = "/mnt/tqsang/chicken_part1/part1_cropped_frames/00045294_L.png"
# input_image_path = "/mnt/tqsang/chicken_part1/part1_cropped_frames/00045294_R.png"
input_image_path = "/mnt/tqsang/chicken_part2/part2_cropped_frames/00005182_R.png"

visualize_blue_mask(input_image_path)
