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

# Function to perform inference and draw bounding boxes
def perform_inference_and_draw_boxes(model, img, output_frame, color, part_name, ii, conf_threshold=0.3, show_score=False):
    results = model.predict(img, conf=conf_threshold)
    part_count = 0
    
    # Draw bounding boxes for the detected objects
    for i, (box, score, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
        box_x1, box_y1, box_x2, box_y2 = box.tolist()
        if score >= conf_threshold:
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
    # text_position = (10, 30 + 40 * ii)  # Update text position for each part
    # cv2.putText(output_frame, f'{part_name}: {part_count}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return part_count

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

        # Put text on the main area avg feather count, avg feather on skin count, avg wing count, avg skin count, avg flesh count
        cv2.putText(output_frame, f"Avg feather count: {round(featheravg_count, 2)}", (x, y-210), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[0], 2)
        cv2.putText(output_frame, f"Avg feather on skin count: {round(feather_on_skin_avg, 2)}", (x, y-180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[1], 2)
        cv2.putText(output_frame, f"Avg wing count: {round(wing_avg, 2)}", (x, y-150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[2], 2)
        cv2.putText(output_frame, f"Avg skin count: {round(skin_avg, 2)}", (x, y-120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[3], 2)
        cv2.putText(output_frame, f"Avg flesh count: {round(flesh_avg, 2)}", (x, y-90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[4], 2)
        # Put text on the main on right hand side
        cv2.putText(output_frame, f"Defect count: {defect_count}", (x, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        cv2.putText(output_frame, f"Normal count: {normal_count}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        #  draw a black box with opacity of 80% right next to the main area to display counts later
        cv2.rectangle(output_frame, (x2 + 20, y), (x2 + 500, y2), (0,0,0), -1)


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

                # Inference for each parts here 
                part_counts = {}

                # Perform inference and draw boxes for each part using the fixed colors and respective confidence thresholds
                for i, part in enumerate(part_config):
                    color = colors[i % len(colors)]  # Cycle through the fixed color list
                    conf_threshold = part_config[part]['conf_threshold']
                    part_count = perform_inference_and_draw_boxes(models[part], img, draw_frame, color, part, i, conf_threshold, show_scores)
                    part_counts[part] = part_count
                    # Put text in the black box with the part counts using the same color
                    cv2.putText(output_frame, f'{part}: {part_count}', (x2 + 20, y + 45 * i + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

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

                        del chicken_dict[chicken_id]


                color = (255, 0, 0) if predicted_class == 'defect' else (0, 0, 255)
                cv2.rectangle(output_frame, (int(box_x1+x), int(box_y1+y)), (int(box_x2+x), int(box_y2+y)), color, 2)
                # Display text on top left of the box
                cv2.putText(output_frame, predicted_class, (int(box_x1+x), int(box_y1+y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)




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
output_directory = '/mnt/tqsang/dot2_test/test_output_dot4'
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
            process_video(model, video, f'/mnt/tqsang/dot2_test/test_output_dot4/{video.split("/")[-1]}', areas[group])
