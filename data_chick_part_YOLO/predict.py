import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from PIL import ImageOps
import numpy as np
import cv2

def process_cropped_frames(model, input_folder, output_folder, max_frame, start_frame):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_folder,"txt")).mkdir(parents=True, exist_ok=True)

    for frame_number in range(start_frame, max_frame + 200000):
        for side in ['L', 'R']:
            input_file = os.path.join(input_folder, f"{frame_number:08d}_{side}.png")

            if not os.path.exists(input_file):
                break

            # Load the image
            img = Image.open(input_file)

            # Run YOLOv8 inference on the image, confidence threshold for detection = 0.9
            results = model.predict(img, conf=0.85) #part2 = 0.8, part1 = 0.85

            # Check if there are detections
            if len(results[0].boxes) > 0:
                img_np = np.array(img)
                hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

                # Define the range for blue color in HSV
                lower_blue = np.array([100, 120, 120]) # part1
                # lower_blue = np.array([100, 60, 60]) # part2

                upper_blue = np.array([140, 255, 255])
                # Threshold the HSV image to get only blue colors
                blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

                # Check if there are blueish pixels in the detection box area
                contains_blue = False
                for box in results[0].boxes.xywh:
                    x, y, w, h = box
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)
                    roi = blue_mask[y1:y2, x1:x2]
                    if np.sum(roi) > 20:
                        contains_blue = True
                        break

                if contains_blue:
                    continue
                
                # Image.fromarray(blue_mask).save(os.path.join(output_folder, f"{frame_number:08d}_{side}_blue_{np.sum(roi)}.png"))

                # Save the results as image
                annotated_img = results[0].plot()
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                Image.fromarray(annotated_img_rgb).save(os.path.join(output_folder, f"{frame_number:08d}_{side}.png"))

                # Save the results as YOLO data format in txt file
                with open(os.path.join(output_folder, f"txt/{frame_number:08d}_{side}.txt"), "w") as f:
                    for box in results[0].boxes:
                        # import pdb;pdb.set_trace()
                        x, y, w, h = box.xywh.cpu().numpy().tolist()[0]
                        f.write(f"{int(box.cls.cpu().numpy().tolist()[0])} {x} {y} {w} {h}\n")



# Load the YOLOv8 model
model = YOLO('/mnt/tqsang/data_chick_part_YOLO/runs/detect/yolov8x_default_rot/weights/best.pt')

max_frame1 = 44606 + 10000
max_frame2 = 4893 + 10000

input_folder_part1 = "/mnt/tqsang/chicken_part1/part1_cropped_frames"
input_folder_part2 = "/mnt/tqsang/chicken_part2/part2_cropped_frames"

output_folder_part1 = "/mnt/tqsang/data_chick_part_YOLO/part1"
output_folder_part2 = "/mnt/tqsang/data_chick_part_YOLO/part2"

process_cropped_frames(model, input_folder_part1, output_folder_part1, max_frame1, max_frame1)
# process_cropped_frames(model, input_folder_part2, output_folder_part2, max_frame2, max_frame2)
