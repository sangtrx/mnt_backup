import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from PIL import ImageOps
import numpy as np
import cv2


# Load the YOLOv8 model
model = YOLO('/mnt/tqsang/hand_dataset_YOLO/runs/detect/yolov8x_hand_full_aug_v3_256_bs16/weights/best.pt')

model.predict(source = '/mnt/tqsang/part2-test.mp4', save=True, imgsz=256, conf=0.5)

