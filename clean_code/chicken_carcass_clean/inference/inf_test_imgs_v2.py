'add mask_carcass to check if the box contains carcass pixels'
import cv2
import glob
import os
import numpy as np
from ultralytics import YOLO

# Function to perform inference and draw bounding boxes
def perform_inference_and_draw_boxes(model, img, output_frame, color, part_name, ii, conf_threshold=0.3, show_score=False):
    results = model.predict(img, conf=conf_threshold)
    part_count = 0
    
    # get the red channel of img, convert to gray then get the biggest area, which is the carcass, using thresholding
    red_channel = img.copy()
    red_channel[:, :, 0] = 0 # set blue and green channels to 0
    red_channel[:, :, 1] = 0
    img_gray = cv2.cvtColor(red_channel, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
    dilated=cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1)))
    contours,_ = cv2.findContours(dilated,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    carcass = max(contours, key=cv2.contourArea)
    # fill carcass with white
    mask_carcass = np.zeros(img_gray.shape, np.uint8)
    cv2.drawContours(mask_carcass, [carcass], -1, 255, -1)


    # save mask_carcass
    # cv2.imwrite('mask_carcass.png', mask_carcass)



    # Draw bounding boxes for the detected objects, only draw and count the boxes that contain the carcass
    for i, (box, score, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
        box_x1, box_y1, box_x2, box_y2 = box.tolist()
        if score >= conf_threshold:
            if box_contains_carcass(mask_carcass, box):
                cv2.rectangle(output_frame, 
                              (int(box_x1), int(box_y1)), 
                              (int(box_x2), int(box_y2)), 
                              color, 2)
                if show_score:
                    text = f'{score:.2f}' # {part_name}: 
                    text_position = (int(box_x1), int(box_y1) - 10)
                    cv2.putText(output_frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                part_count += 1

    # Put text next to the image with the part counts using the same color
    text_position = (10, 30 + 40 * ii)  # Update text position for each part
    cv2.putText(output_frame, f'{part_name}: {part_count}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return part_count

# Function to check there is carcass pixels in the box
def box_contains_carcass(mask_carcass, box):
    box_x1, box_y1, box_x2, box_y2 = box.tolist()
    box_x1, box_y1, box_x2, box_y2 = int(box_x1), int(box_y1), int(box_x2), int(box_y2)
    box_mask = np.zeros(mask_carcass.shape, np.uint8)
    box_mask[box_y1:box_y2, box_x1:box_x2] = mask_carcass[box_y1:box_y2, box_x1:box_x2]
    return np.any(box_mask)



# Define the paths to the YOLO models and confidence thresholds for each part
part_config = {
    'feather': {'model_path': '/home/tqsang/JSON2YOLO/feather_YOLO_det/runs/detect/yolov8x_dot3_noRot/weights/best.pt', 'conf_threshold': 0.2},
    'feather on skin': {'model_path': '/home/tqsang/JSON2YOLO/feather_on_skin_YOLO_det/runs/detect/yolov8x_feather_on_skin/weights/best.pt', 'conf_threshold': 0.2},
    'wing': {'model_path': '/home/tqsang/JSON2YOLO/wing_YOLO_det/runs/detect/yolov8x_wing_dot3_nohardcase_noMosaic_noScale/weights/best.pt', 'conf_threshold': 0.2},
    'skin': {'model_path': '/home/tqsang/JSON2YOLO/skin_YOLO_det/runs/detect/yolov8x_skin/weights/best.pt', 'conf_threshold': 0.5},
    'flesh': {'model_path': '/home/tqsang/JSON2YOLO/flesh_YOLO_det/runs/detect/yolov8x_flesh_dot3/weights/best.pt', 'conf_threshold': 0.5}
}

# Load the YOLO models
models = {part: YOLO(part_config[part]['model_path']) for part in part_config}

# Define a fixed color list
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 255), (50, 100, 255), (0, 255, 255)]

# Create the output folder if it doesn't exist
output_directory = '/mnt/tqsang/dot2_test/test_output_dot3'
os.makedirs(output_directory, exist_ok=True)

# Toggle for showing scores
show_scores = True  # Change this to False if you want to hide scores

# Loop through images in the specified directory
image_directory = '/mnt/tqsang/dot2_test/test_imgs'

for image_path in glob.glob(os.path.join(image_directory, '*.png')):
    img = cv2.imread(image_path)
    output_frame = img.copy()

    part_counts = {}

    # Perform inference and draw boxes for each part using the fixed colors and respective confidence thresholds
    for i, part in enumerate(part_config):
        color = colors[i % len(colors)]  # Cycle through the fixed color list
        conf_threshold = part_config[part]['conf_threshold']
        part_count = perform_inference_and_draw_boxes(models[part], img, output_frame, color, part, i, conf_threshold, show_scores)
        part_counts[part] = part_count

    # Save the output image to the output directory
    output_image_path = os.path.join(output_directory, os.path.basename(image_path))
    cv2.imwrite(output_image_path, output_frame)

print("Processing complete.")
