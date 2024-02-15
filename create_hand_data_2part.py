import os
import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import copyfile
import math
import cv2
def convert_pascal_voc_to_yolo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    
    objects = root.findall("object")
    yolo_annotations = []
    
    for obj in objects:
        class_id = 0  # Change the class of all XML to 0 in YOLO
        bbox = obj.find("bndbox")
        
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        
        # Check if the coordinates of the bbox are out of the frame
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width, xmax)
        ymax = min(height, ymax)
        
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height
        
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")
        
    return yolo_annotations

parts1 = [('part1', '/mnt/tqsang/hand_only_part1', '/mnt/tqsang/chicken_part1/frames')]

parts2 = [('part2', '/mnt/tqsang/hand_only_part2', '/mnt/tqsang/part2.mp4')]

output_folder = "/mnt/tqsang/hand_dataset_YOLO"

for part, source_folder, image_folder in parts1:

    xml_files = [file for file in os.listdir(source_folder) if file.endswith(".xml")]
    n_train = math.floor(len(xml_files) * 0.8)
    n_val = len(xml_files) - n_train

    for i, xml_file in enumerate(xml_files):
        xml_path = os.path.join(source_folder, xml_file)
        image_file = xml_file[:-4] + ".png"
        # image_path = os.path.join(image_folder, image_file)
        file_num = int(xml_file[:-4])  # remove '.xml' and convert to integer
        file_num -= 1  # subtract 1
        formatted_num = f"{file_num:08d}"         
        image_path = os.path.join(image_folder, formatted_num + ".png") # back 1 frame
        
        yolo_annotations = convert_pascal_voc_to_yolo(xml_path)
        
        if i < n_train:
            target_folder = os.path.join(output_folder, "train")
        else:
            target_folder = os.path.join(output_folder, "val")
            
        target_image_folder = os.path.join(target_folder, "images")
        target_label_folder = os.path.join(target_folder, "labels")
        
        Path(target_image_folder).mkdir(parents=True, exist_ok=True)
        Path(target_label_folder).mkdir(parents=True, exist_ok=True)
        
        # Copy the image
        copyfile(image_path, os.path.join(target_image_folder, f"{part}_{image_file}"))
        
        # Save the YOLO annotations
        with open(os.path.join(target_label_folder, f"{part}_{image_file[:-4]}.txt"), "w") as f:
            for annotation in yolo_annotations:
                f.write(annotation + "\n")

for part, source_folder, video_path in parts2:

    xml_files = [file for file in os.listdir(source_folder) if file.endswith(".xml")]
    n_train = math.floor(len(xml_files) * 0.8)
    n_val = len(xml_files) - n_train

    video = cv2.VideoCapture(video_path)

    for i, xml_file in enumerate(sorted(xml_files)):
        xml_path = os.path.join(source_folder, xml_file)
        file_num = int(xml_file[:-4])  # remove '.xml' and convert to integer

        # Read frames from video until the frame number matches with the XML file
        while True:
            ret, frame = video.read()
            if not ret:
                break
            current_frame_num = int(video.get(cv2.CAP_PROP_POS_FRAMES))

            if current_frame_num == file_num:
                break

        yolo_annotations = convert_pascal_voc_to_yolo(xml_path)
        
        if i < n_train:
            target_folder = os.path.join(output_folder, "train")
        else:
            target_folder = os.path.join(output_folder, "val")
            
        target_image_folder = os.path.join(target_folder, "images")
        target_label_folder = os.path.join(target_folder, "labels")
        
        Path(target_image_folder).mkdir(parents=True, exist_ok=True)
        Path(target_label_folder).mkdir(parents=True, exist_ok=True)

        # Convert the frame to an image file (PNG)
        image_file = f"{part}_{xml_file[:-4]}.png"
        cv2.imwrite(os.path.join(target_image_folder, image_file), frame)
        
        # Save the YOLO annotations
        with open(os.path.join(target_label_folder, f"{part}_{xml_file[:-4]}.txt"), "w") as f:
            for annotation in yolo_annotations:
                f.write(annotation + "\n")

    video.release()