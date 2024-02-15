'process feather and feather on skin together, add IOU and blackish pixels check'
import cv2
import glob
import os
import numpy as np
from ultralytics import YOLO
import ultralytics
import torch
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

def FEATHER_perform_inference_and_draw_boxes(img, output_frame):
    # Combine results of 'feather' and 'feather on skin'
    feather_results = models['feather'].predict(img, conf=part_config['feather']['conf_threshold'])
    feather_on_skin_results = models['feather on skin'].predict(img, conf=part_config['feather on skin']['conf_threshold'])

    # Create new Results object class to store the combined results
    class Results:
        xyxy = []
        conf = []
        cls = []

    # Create a new simple results object to store the combined results in numpy
    combined_results = Results()
    combined_results.xyxy = np.concatenate((feather_results[0].boxes.xyxy.cpu().numpy(), feather_on_skin_results[0].boxes.xyxy.cpu().numpy()))
    combined_results.conf = np.concatenate((feather_results[0].boxes.conf.cpu().numpy(), feather_on_skin_results[0].boxes.conf.cpu().numpy()))
    combined_results.cls = np.concatenate((feather_results[0].boxes.cls.cpu().numpy(), feather_on_skin_results[0].boxes.cls.cpu().numpy()))

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


    # Check if the results are on the carcass or not
    i = 0
    while i < len(combined_results.xyxy):
        box = combined_results.xyxy[i]
        score = combined_results.conf[i]
        if score >= part_config['feather']['conf_threshold'] or score >= part_config['feather on skin']['conf_threshold']:
            if box_contains_carcass(mask_carcass, box):
                # Keep the box
                i += 1
            else:
                # Remove the box
                combined_results.xyxy = np.delete(combined_results.xyxy, i, 0)
                combined_results.conf = np.delete(combined_results.conf, i, 0)
                combined_results.cls = np.delete(combined_results.cls, i, 0)
    
    # function to calculate IOU
    def IOU(box1, box2):
        box1_x1, box1_y1, box1_x2, box1_y2 = box1.tolist()
        box2_x1, box2_y1, box2_x2, box2_y2 = box2.tolist()
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        union_area = box1_area + box2_area - intersection_area
        return intersection_area / union_area
        
    # loop through the combined results if 2 boxes have IOU > threshold, keep the one bigger area
    i = 0
    while i < len(combined_results.xyxy):
        box1 = combined_results.xyxy[i]
        j = i + 1
        while j < len(combined_results.xyxy):
            box2 = combined_results.xyxy[j]
            if IOU(box1, box2) > 0.2: # threshold
                # Remove the smaller box
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                if area1 > area2:
                    combined_results.xyxy = np.delete(combined_results.xyxy, j, 0)
                    combined_results.conf = np.delete(combined_results.conf, j, 0)
                    combined_results.cls = np.delete(combined_results.cls, j, 0)
                else:
                    combined_results.xyxy = np.delete(combined_results.xyxy, i, 0)
                    combined_results.conf = np.delete(combined_results.conf, i, 0)
                    combined_results.cls = np.delete(combined_results.cls, i, 0)
                    i -= 1
                    break
            else:
                j += 1
        i += 1

    count_feather = 0
    count_feather_on_skin = 0   
    # Loop through the combine results to classify based on the presence of blackish pixels
    for i, (box, score, cls) in enumerate(zip(combined_results.xyxy, combined_results.conf, combined_results.cls)):
        box_x1, box_y1, box_x2, box_y2 = box.tolist()
        if score >= part_config['feather']['conf_threshold'] or score >= part_config['feather on skin']['conf_threshold']:
            # count blackish pixels in the box in img
            box_img = img[int(box_y1):int(box_y2), int(box_x1):int(box_x2)]
            box_img_gray = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)
            _, box_img_thresh = cv2.threshold(box_img_gray, 20, 255, cv2.THRESH_BINARY)
            blackish_pixels = np.sum(box_img_thresh == 0)

            # Determine if it's feather or feather on skin based on the presence of blackish pixels in the box
            if blackish_pixels > 0:
                part = 'feather'
            else:
                part = 'feather on skin'

            # Draw bounding boxes for the detected objects with their colors
            color = colors[1] if part == 'feather on skin' else colors[0]
            cv2.rectangle(output_frame, 
                            (int(box_x1), int(box_y1)), 
                            (int(box_x2), int(box_y2)), 
                            color, 2)
            
            # count each part
            if part == 'feather':
                count_feather += 1
            else:
                count_feather_on_skin += 1

    # Put text next to the image with the part counts using the same color
    text_position = (10, 30 + 40 * 0)  # Update text position for each part
    cv2.putText(output_frame, f'feather: {count_feather}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, colors[0], 2)       
    text_position = (10, 30 + 40 * 1)  # Update text position for each part
    cv2.putText(output_frame, f'feather on skin: {count_feather_on_skin}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, colors[1], 2)    


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

    # Perform inference and draw boxes for each part using the fixed colors and respective confidence thresholds, do the feather' and 'feather on skin' separately
    part_count = FEATHER_perform_inference_and_draw_boxes(img, output_frame)
    # part_counts[part] = part_count
    # do the rest parts
    for i, part in enumerate(['wing', 'skin', 'flesh']):
        i += 2
        color = colors[i % len(colors)]  # Cycle through the fixed color list
        conf_threshold = part_config[part]['conf_threshold']
        part_count = perform_inference_and_draw_boxes(models[part], img, output_frame, color, part, i, conf_threshold, show_scores)
        part_counts[part] = part_count

    # Save the output image to the output directory
    output_image_path = os.path.join(output_directory, os.path.basename(image_path))
    cv2.imwrite(output_image_path, output_frame)

print("Processing complete.")
