import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time

# Initialize and display "Init..." text on the frame while initializing scale_areas
def init_scale_areas(scale_model, video_capture):
    scale_areas = []

    # Text properties for initialization message
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    font_scale = 1
    color = (0, 0, 255)
    thickness = 2
    init_text = "Init..."

    # Initialize the timer for the "Init..." text
    init_start_time = time.time()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Display "Init..." text
        cv2.putText(frame, init_text, org, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Initialization', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Detect scales using the scale_model
        results = scale_model.predict(frame, conf=0.8)

        # Assuming there are 2 scales detected
        if len(results[0]) == 2:
            scale_areas = []
            for i, (box, score, class_) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
                x1, y1, x2, y2 = box.tolist()
                # Convert coordinates to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Ensure the scale area is square
                width = x2 - x1
                height = y2 - y1
                if width > height:
                    # Extend the height to make it square
                    y2 += (width - height) // 2
                    y1 -= (width - height) // 2
                else:
                    # Extend the width to make it square
                    x2 += (height - width) // 2
                    x1 -= (height - width) // 2

                # Append the detected scale area to the list
                scale_areas.append((x1, y1, x2, y2))

            break

    return scale_areas

# Process the video frame by frame
def process_video(model, hand_model, cap, areas, x1_hand, y1_hand, x2_hand, y2_hand):
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        output_frame = frame.copy()

        # Hand detection
        hand_area = frame[y1_hand:y2_hand, x1_hand:x2_hand]
        hand_results = hand_model.predict(hand_area, conf=0.5)

        if len(hand_results[0].boxes) > 0:
            annotated_hand_area = hand_results[0].plot()
            output_frame[y1_hand:y2_hand, x1_hand:x2_hand] = annotated_hand_area
            cv2.rectangle(output_frame, (x1_hand, y1_hand), (x2_hand, y2_hand), (255, 0, 0), 2)
            cv2.imshow('Video', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Chicken Part detection
        for idx, area in enumerate(areas):
            x1, y1, x2, y2 = area
            cropped_frame = frame[y1:y2, x1:x2]

            results = model.predict(cropped_frame, conf=0.8)

            if len(results[0].boxes) > 0:
                annotated_frame = results[0].plot()
                output_frame[y1:y2, x1:x2] = annotated_frame

            color = (0, 255, 0) if idx == 0 else (0, 0, 255)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow('Video', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Paths to YOLO model weights
model_path = '../chicken_model/weights/best.pt'
hand_model_path = '../hand_model/weights/best.pt'
scale_model_path = '../scale_model/scale_v2_ShiftScaleRotate/weights/best.pt'

# Load YOLO models
model = YOLO(model_path)
hand_model = YOLO(hand_model_path)
scale_model = YOLO(scale_model_path)

# Change to read video directly from the virtual camera device (e.g., camera index 2)
CAM_NUM = 2
video_capture = cv2.VideoCapture(CAM_NUM)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize the scale_areas_part in 5 seconds
scale_areas = init_scale_areas(scale_model, video_capture)

# Calculate bounding rectangle coordinates for the scale_areas
x1 = min(area[0] for area in scale_areas)
y1 = min(area[1] for area in scale_areas)
x2 = max(area[2] for area in scale_areas)
y2 = max(area[3] for area in scale_areas)

# Ensure 16:9 aspect ratio for the rectangle
rect_width = x2 - x1
rect_height = y2 - y1
aspect_ratio = rect_width / rect_height

if aspect_ratio < 16 / 9:
    rect_width = int(rect_height * 16 / 9)
else:
    rect_height = int(rect_width * 9 / 16)

x1_hand = max(0, x1 - (rect_width - (x2 - x1)) // 2)
y1_hand = max(0, y1 - (rect_height - (y2 - y1)) // 2)
x2_hand = min(width, x1 + rect_width)
y2_hand = min(height, y1 + rect_height)

# Process the video with detected scale areas and hand region
process_video(model, hand_model, video_capture, scale_areas, x1_hand, y1_hand, x2_hand, y2_hand)

