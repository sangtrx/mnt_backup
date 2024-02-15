## vid_v2  = imgs_v5

import cv2
import numpy as np
import glob
from ultralytics import YOLO
import os 
from collections import Counter


# Global counter for normal and defect chickens
defect_count = 0
normal_count = 0

# Dictionary to keep track of each chicken
chicken_dict = {}

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

# Create new Results object class to store the results
class Results:
    xyxy = []
    conf = []
    cls = []
# Function to perform inference and draw bounding boxes
def perform_inference_and_draw_boxes(model, img, output_frame, color, part_name, ii, conf_threshold=0.3, show_score=False, mask_carcass = None):
    results = model.predict(img, conf=conf_threshold)
    part_count = 0

    # Create a new simple results object to store the combined results in numpy
    combined_results = Results()
    combined_results.xyxy = results[0].boxes.xyxy.cpu().numpy()
    combined_results.conf = results[0].boxes.conf.cpu().numpy()
    combined_results.cls = results[0].boxes.cls.cpu().numpy()


    # Check if the results are on the carcass or not
    i = 0
    while i < len(combined_results.xyxy):
        box = combined_results.xyxy[i]
        score = combined_results.conf[i]
        # if score >= part_config['feather']['conf_threshold'] :
        if box_contains_carcass(mask_carcass, box):
            # Keep the box
            i += 1
        else:
            # Remove the box
            combined_results.xyxy = np.delete(combined_results.xyxy, i, 0)
            combined_results.conf = np.delete(combined_results.conf, i, 0)
            combined_results.cls = np.delete(combined_results.cls, i, 0)
    

        
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

    # Draw bounding boxes for the detected objects, only draw and count the boxes that contain the carcass
    for i, (box, score, cls) in enumerate(zip(combined_results.xyxy, combined_results.conf, combined_results.cls)):
        box_x1, box_y1, box_x2, box_y2 = box.tolist()
        cv2.rectangle(output_frame, 
                        (int(box_x1), int(box_y1)), 
                        (int(box_x2), int(box_y2)), 
                        color, 2)
        if show_score:
            text = f'{score:.2f}' # {part_name}: 
            text_position = (int(box_x1), int(box_y1) - 10)
            cv2.putText(output_frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        part_count += 1


    return part_count

# Function to check there is carcass pixels in the box
def box_contains_carcass(mask_carcass, box):
    box_x1, box_y1, box_x2, box_y2 = box.tolist()
    box_x1, box_y1, box_x2, box_y2 = int(box_x1), int(box_y1), int(box_x2), int(box_y2)
    box_mask = np.zeros(mask_carcass.shape, np.uint8)
    box_mask[box_y1:box_y2, box_x1:box_x2] = mask_carcass[box_y1:box_y2, box_x1:box_x2]
    return np.any(box_mask)

def FEATHER_perform_inference_and_draw_boxes(img, output_frame, mask_carcass):
    # Combine results of 'feather' and 'feather on skin'
    feather_results = models['feather'].predict(img, conf=part_config['feather']['conf_threshold'])
    feather_on_skin_results = models['feather on skin'].predict(img, conf=part_config['feather on skin']['conf_threshold'])


    # Create a new simple results object to store the combined results in numpy
    combined_results = Results()
    combined_results.xyxy = np.concatenate((feather_results[0].boxes.xyxy.cpu().numpy(), feather_on_skin_results[0].boxes.xyxy.cpu().numpy()))
    combined_results.conf = np.concatenate((feather_results[0].boxes.conf.cpu().numpy(), feather_on_skin_results[0].boxes.conf.cpu().numpy()))
    combined_results.cls = np.concatenate((feather_results[0].boxes.cls.cpu().numpy(), feather_on_skin_results[0].boxes.cls.cpu().numpy()))


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
    feather_list = []
    feather_on_skin_list = []
    # Loop through the combine results to classify based on the presence of blackish pixels
    for i, (box, score, cls) in enumerate(zip(combined_results.xyxy, combined_results.conf, combined_results.cls)):
        box_x1, box_y1, box_x2, box_y2 = box.tolist()
        if score >= part_config['feather']['conf_threshold'] or score >= part_config['feather on skin']['conf_threshold']:
            # count blackish pixels in the box in img
            box_img = img[int(box_y1):int(box_y2), int(box_x1):int(box_x2)]
            box_img_gray = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)
            _, box_img_thresh = cv2.threshold(box_img_gray, 50, 255, cv2.THRESH_BINARY)
            blackish_pixels = np.sum(box_img_thresh == 0)

            # Determine if it's feather or feather on skin based on the presence of blackish pixels in the box
            if blackish_pixels > 0:
                part = 'feather'
            else:
                part = 'feather on skin'

            # add to list
            if part == 'feather':
                feather_list.append(i)
            else:
                feather_on_skin_list.append(i)
    
    feather_mask = np.zeros(img.shape, np.uint8)
    feather_on_skin_mask = np.zeros(img.shape, np.uint8)
    # loop through each list to fill the boxes with white to create masks 
    for i in feather_list:
        box = combined_results.xyxy[i]
        box_x1, box_y1, box_x2, box_y2 = box.tolist()
        cv2.rectangle(feather_mask, 
                              (int(box_x1), int(box_y1)), 
                              (int(box_x2), int(box_y2)), 
                              (255, 255, 255), -1)
    for i in feather_on_skin_list:      
        box = combined_results.xyxy[i]
        box_x1, box_y1, box_x2, box_y2 = box.tolist()
        cv2.rectangle(feather_on_skin_mask, 
                              (int(box_x1), int(box_y1)), 
                              (int(box_x2), int(box_y2)), 
                              (255, 255, 255), -1)
    # find contours of the masks, number of contours = number of parts, draw contours on output_frame 
    feather_mask_gray = cv2.cvtColor(feather_mask, cv2.COLOR_BGR2GRAY)
    _, feather_mask_thresh = cv2.threshold(feather_mask_gray, 20, 255, cv2.THRESH_BINARY)  
    feather_on_skin_mask_gray = cv2.cvtColor(feather_on_skin_mask, cv2.COLOR_BGR2GRAY)
    _, feather_on_skin_mask_thresh = cv2.threshold(feather_on_skin_mask_gray, 20, 255, cv2.THRESH_BINARY)
    feather_contours,_ = cv2.findContours(feather_mask_thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    feather_on_skin_contours,_ = cv2.findContours(feather_on_skin_mask_thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output_frame, feather_contours, -1, colors[0], 2)
    cv2.drawContours(output_frame, feather_on_skin_contours, -1, colors[1], 2)

    count_feather = len(feather_contours)
    count_feather_on_skin = len(feather_on_skin_contours)
 

    return count_feather, count_feather_on_skin


######################
def process_video(model, video_path, output_path, area):
    global defect_count, normal_count, chicken_dict
    # Define a counter for unique chicken identifiers
    chicken_id_counter = 0
    hit_left = False
    hit_right = False
    chicken_dict = {}
    defect_count = 0
    normal_count = 0
    featheravg_count = 0
    feather_on_skin_avg = 0
    skin_avg = 0
    wing_avg = 0
    flesh_avg = 0
    last_chick = 0
    avg_result = 0
    frame_chick = 0
    

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        frame_count += 1

        if not success:
            break

        output_frame = frame.copy()


        # Define the bounding main area
        x, y, x2, y2 = area

        # Crop the frame to main area
        cropped_frame = frame[y:y2, x:x2]

        # Perform inference on the main area
        results = model.predict(cropped_frame, conf=0.8)

        # draw a black box with opacity of 80% on top of the main area to display counts later
        cv2.rectangle(output_frame, (x, y-500), (x2, y), (0,0,0), -1)
        cv2.putText(output_frame, f"Object based", (x, y-450), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255,255,255), 3)
        # put text last_chick and its type
        cv2.putText(output_frame, f"Last carcass: {last_chick}", (x, y-400), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255,255,255), 2)
        # Put text on the main area avg feather count, avg feather on skin count, avg wing count, avg skin count, avg flesh count
        cv2.putText(output_frame, f"Avg feather around count: {round(featheravg_count, 2)}", (x, y-350), cv2.FONT_HERSHEY_SIMPLEX, 1.7, colors[0], 2)
        cv2.putText(output_frame, f"Avg feather on skin count: {round(feather_on_skin_avg, 2)}", (x, y-300), cv2.FONT_HERSHEY_SIMPLEX, 1.7, colors[1], 2)
        cv2.putText(output_frame, f"Avg wing count: {round(wing_avg, 2)}", (x, y-250), cv2.FONT_HERSHEY_SIMPLEX, 1.7, colors[2], 2)
        cv2.putText(output_frame, f"Avg skin count: {round(skin_avg, 2)}", (x, y-200), cv2.FONT_HERSHEY_SIMPLEX, 1.7, colors[3], 2)
        cv2.putText(output_frame, f"Avg flesh count: {round(flesh_avg, 2)}", (x, y-150), cv2.FONT_HERSHEY_SIMPLEX, 1.7, colors[4], 2)
        # Put text on the main on right hand side
        cv2.putText(output_frame, f"Defect count: {defect_count}", (x, y-100), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0,0,255), 2)
        cv2.putText(output_frame, f"Normal count: {normal_count}", (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0,255,0), 2)


        #  draw a black box with opacity of 80% right next to the main area to display counts later
        cv2.rectangle(output_frame, (x2 + 20, y), (x2 + 650, y+450), (0,0,0), -1)


        # Iterate through the detections
        for i, (box, score, class_) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
            box_x1, box_y1, box_x2, box_y2 = box.tolist()
            # Check if the box is completely inside the main area
            if (box_x1 >= 5 and box_x2 <= x2-x-5): # Only need to check 2left n right vertical line 

                # Calculate the smallest square area that contains the bounding box
                min_side_length = max(box_x2 - box_x1, box_y2 - box_y1)
                center_x = (box_x1 + box_x2) / 2
                center_y = (box_y1 + box_y2) / 2
                square_x1 = int(center_x - min_side_length / 2)
                square_y1 = int(center_y - min_side_length / 2)
                square_x2 = int(center_x + min_side_length / 2)
                square_y2 = int(center_y + min_side_length / 2)

                # Adjust coordinates to stay within the valid range of the cropped_frame
                if square_x1 < 0:
                    square_x2 -= square_x1
                    square_x1 = 0
                if square_y1 < 0:
                    square_y2 -= square_y1
                    square_y1 = 0
                if square_x2 > cropped_frame.shape[1]:
                    square_x1 -= (square_x2 - cropped_frame.shape[1])
                    square_x2 = cropped_frame.shape[1]
                if square_y2 > cropped_frame.shape[0]:
                    square_y1 -= (square_y2 - cropped_frame.shape[0])
                    square_y2 = cropped_frame.shape[0]


                # Crop the smallest square area from main frame
                cropped_square = cropped_frame[square_y1:square_y2, square_x1:square_x2]
                img = cropped_square.copy()
                draw_frame = cropped_square.copy()

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

                # # Put text for carcass count and frame number for that carcass
                # cv2.putText(output_frame, f"Carcass number: {last_chick}", (x2 + 30, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)

                # Inference for each parts here 
                part_counts = {}

                # Perform inference for feather and feather on skin
                feather_count, feather_on_skin_count = FEATHER_perform_inference_and_draw_boxes(img, draw_frame, mask_carcass)
                part_counts['feather'] = feather_count
                part_counts['feather on skin'] = feather_on_skin_count
                # Put text in the black box with the part counts using the same color
                cv2.putText(output_frame, f'feather around: {feather_count}', (x2 + 20, y + 50 * 3 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.7, colors[0], 2)
                cv2.putText(output_frame, f'feather on skin: {feather_on_skin_count}', (x2 + 20, y + 50 * 4 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.7, colors[1], 2)

                # Perform inference and draw boxes for each part using the fixed colors and respective confidence thresholds
                for i, part in enumerate(['wing', 'skin', 'flesh']):
                    i += 2  # Start from 2 to skip 'feather' and 'feather on skin'
                    color = colors[i % len(colors)]  # Cycle through the fixed color list
                    conf_threshold = part_config[part]['conf_threshold']
                    part_count = perform_inference_and_draw_boxes(models[part], img, draw_frame, color, part, i, conf_threshold, show_scores, mask_carcass)
                    part_counts[part] = part_count
                    # Put text in the black box with the part counts using the same color
                    cv2.putText(output_frame, f'{part}: {part_count}', (x2 + 20, y + 50 * (i+3) + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.7, color, 2)

                # paste draw_frame back onto output_frame  
                output_frame[y + square_y1:y + square_y2, x + square_x1: x + square_x2] = draw_frame   

                # Get class prediction for all the parts, if one of them > 1 then it's defect
                feather_count = part_counts['feather']
                feather_on_skin_count = part_counts['feather on skin']
                wing_count = part_counts['wing']
                skin_count = part_counts['skin']
                flesh_count = part_counts['flesh']
                if feather_count >= 1 or feather_on_skin_count >= 1 or wing_count >= 1 or skin_count >= 1 or flesh_count >= 1:
                    predicted_class = 'defect'
                else:
                    predicted_class = 'normal'



                # Check if chicken is crossing the area of left line
                if    20 >= box_x1 >= 5:
                    if hit_left == False:
                        hit_left = True  
                        hit_right = False                  
                        # Create a unique chicken id
                        chicken_id = f"{chicken_id_counter}"
                        chicken_id_counter += 1
                        # update frame_chick
                        frame_chick += 1
                        # Add the chicken to the dictionary with all parts and their counts
                        chicken_dict[chicken_id] = {"classifications": [predicted_class], "feather_count":[feather_count], "feather_on_skin_count":[feather_on_skin_count], "wing_count":[wing_count], "skin_count":[skin_count], "flesh_count":[flesh_count]}

                # If chicken already crossed the left line, continue to update its classifications
                if  hit_left == True and hit_right == False:
                        chicken_dict[chicken_id]["classifications"].append(predicted_class)
                        chicken_dict[chicken_id]["feather_count"].append(feather_count)
                        chicken_dict[chicken_id]["feather_on_skin_count"].append(feather_on_skin_count)
                        chicken_dict[chicken_id]["wing_count"].append(wing_count)
                        chicken_dict[chicken_id]["skin_count"].append(skin_count)
                        chicken_dict[chicken_id]["flesh_count"].append(flesh_count)
                        # update frame_chick
                        frame_chick += 1

                # If chicken crossed the area of right line
                if    x2-x-20<= box_x2 <= x2-x-5:
                    if hit_right == False and hit_left == True:
                        hit_right = True
                        hit_left  = False
                        # Compute average classification result
                        avg_result = Counter(chicken_dict[chicken_id]["classifications"]).most_common(1)[0][0]

                        # Calculate average feather count
                        feather_sum = sum(chicken_dict[chicken_id]["feather_count"])
                        feather_total = len(chicken_dict[chicken_id]["feather_count"])
                        featheravg_count = feather_sum / feather_total if feather_total != 0 else 0

                        # Calculate average feather on skin count
                        feather_on_skin_sum = sum(chicken_dict[chicken_id]["feather_on_skin_count"])
                        feather_on_skin_total = len(chicken_dict[chicken_id]["feather_on_skin_count"])
                        feather_on_skin_avg = feather_on_skin_sum / feather_on_skin_total if feather_on_skin_total != 0 else 0

                        # Calculate average wing count
                        wing_sum = sum(chicken_dict[chicken_id]["wing_count"])
                        wing_total = len(chicken_dict[chicken_id]["wing_count"])
                        wing_avg = wing_sum / wing_total if wing_total != 0 else 0

                        # Calculate average skin count
                        skin_sum = sum(chicken_dict[chicken_id]["skin_count"])
                        skin_total = len(chicken_dict[chicken_id]["skin_count"])
                        skin_avg = skin_sum / skin_total if skin_total != 0 else 0

                        # Calculate average flesh count
                        flesh_sum = sum(chicken_dict[chicken_id]["flesh_count"])
                        flesh_total = len(chicken_dict[chicken_id]["flesh_count"])
                        flesh_avg = flesh_sum / flesh_total if flesh_total != 0 else 0


                        # Update the counters and remove the chicken from the dictionary
                        if avg_result == 'defect':
                            defect_count += 1
                        else:
                            normal_count += 1
                        # update last_chick and its type defect/normal
                        last_chick = f'{chicken_id_counter}' + " Type: " + avg_result

                        # reset frame_chick
                        frame_chick = 0
                        
                        del chicken_dict[chicken_id]

                # put text current chicken number and its type, and its frame number
                cv2.putText(output_frame, f'Current carcass: {chicken_id_counter}', (x2 + 20, y + 50 * 0 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255,255,255), 2)
                cv2.putText(output_frame, f'Frame: {frame_chick}', (x2 + 20, y + 50 * 1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255,255,255), 2)
                cv2.putText(output_frame, f'Type: {predicted_class}', (x2 + 20, y + 50 * 2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255,255,255), 2)


                color = (255, 0, 0) if predicted_class == 'defect' else (0, 0, 255)
                cv2.rectangle(output_frame, (int(box_x1+x), int(box_y1+y)), (int(box_x2+x), int(box_y2+y)), color, 2)
                # Display text on top left of the box
                cv2.putText(output_frame, predicted_class, (int(box_x1+x), int(box_y1+y-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)




        # Draw the bounding rectangle area
        color = (0, 255, 0) 
        cv2.rectangle(output_frame, (x, y), (x2, y2), color, 2)

        # write frame
        out.write(output_frame)
    cap.release()
    out.release()

# det model
model_path = '/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/runs/detect/yolov8x_brio_det_only/weights/best.pt'
model = YOLO(model_path)

# Define the paths to the YOLO models and confidence thresholds for each part
part_config = {
    'feather': {'model_path': '/home/tqsang/JSON2YOLO/feather_YOLO_det/runs/detect/yolov8x_dot3_noRot/weights/best.pt', 'conf_threshold': 0.15},
    'feather on skin': {'model_path': '/home/tqsang/JSON2YOLO/feather_on_skin_YOLO_det/runs/detect/yolov8x_feather_on_skin/weights/best.pt', 'conf_threshold': 0.15},
    'wing': {'model_path': '/home/tqsang/JSON2YOLO/wing_YOLO_det/runs/detect/yolov8x_wing_dot3_nohardcase_noMosaic_noScale/weights/best.pt', 'conf_threshold': 0.2},
    'skin': {'model_path': '/home/tqsang/JSON2YOLO/skin_YOLO_det/runs/detect/yolov8x_skin_dot3_noMosaic_add/weights/best.pt', 'conf_threshold': 0.2},
    'flesh': {'model_path': '/home/tqsang/JSON2YOLO/flesh_YOLO_det/runs/detect/yolov8x_flesh_dot3/weights/best.pt', 'conf_threshold': 0.4}
}

# Load the YOLO models
models = {part: YOLO(part_config[part]['model_path']) for part in part_config}

# Define a fixed color list
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 255), (50, 100, 255), (0, 255, 255)]

# Create the output folder if it doesn't exist
output_directory = '/mnt/tqsang/dot2_test/test_dot3_vids'
os.makedirs(output_directory, exist_ok=True)

# Toggle for showing scores
show_scores = False  # Change this to False if you want to hide scores

# Define the areas for each group
areas = {
    '1-13': [1319, 598, 1319+1000, 598+1000],
}

# Get all video files
video_files = glob.glob('/mnt/tqsang/dot2_test/dot2_*.mp4')

# Group video files
video_groups = {
    '1-13': [],
}

for video_file in video_files:
    video_number = int(video_file.split('_')[2].split('.')[0])
    if 1 <= video_number <= 13:
        video_groups['1-13'].append(video_file)

# Process each video
for group, videos in video_groups.items():
    for video in videos:
            print(video)
            process_video(model, video, f'/mnt/tqsang/dot2_test/test_dot3_vids/{video.split("/")[-1]}', areas[group])
            
