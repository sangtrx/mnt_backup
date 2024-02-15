import os
import random
import shutil
import xml.etree.ElementTree as ET

# Paths to the folders
images_folder = '/mnt/tqsang/scale_sample'
labels_folder = '/mnt/tqsang/label_scale_sample'
output_folder = '/mnt/tqsang/yolo_scale'

# Create necessary subdirectories
datasets_folder = os.path.join(output_folder, 'datasets')
train_folder = os.path.join(datasets_folder, 'train')
val_folder = os.path.join(datasets_folder, 'val')
train_images_folder = os.path.join(train_folder, 'images')
train_labels_folder = os.path.join(train_folder, 'labels')
val_images_folder = os.path.join(val_folder, 'images')
val_labels_folder = os.path.join(val_folder, 'labels')

for folder in [datasets_folder, train_folder, val_folder, train_images_folder, train_labels_folder, val_images_folder, val_labels_folder]:
    os.makedirs(folder, exist_ok=True)

# Create a list of XML files
xml_files = [xml_file for xml_file in os.listdir(labels_folder) if xml_file.endswith('.xml')]

# Shuffle the XML files for randomness
random.shuffle(xml_files)

# Calculate train-val split index
split_index = int(0.8 * len(xml_files))

# Iterate through XML files
for i, xml_file in enumerate(xml_files):
    xml_path = os.path.join(labels_folder, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_filename = root.find('filename').text
    image_path = os.path.join(images_folder, image_filename)
    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)

    # Determine destination folders based on the split
    if i < split_index:
        dest_images_folder = train_images_folder
        dest_labels_folder = train_labels_folder
    else:
        dest_images_folder = val_images_folder
        dest_labels_folder = val_labels_folder

    # Copy the image to the appropriate folder
    shutil.copy(image_path, os.path.join(dest_images_folder, image_filename))

    # Process the XML and write YOLO labels
    yolo_label_path = os.path.join(dest_labels_folder, image_filename.replace('.png', '.txt'))
    with open(yolo_label_path, 'w') as yolo_label_file:
        for obj in root.findall('object'):
            class_id = int(obj.find('name').text)
            xmin = float(obj.find('bndbox/xmin').text)
            ymin = float(obj.find('bndbox/ymin').text)
            xmax = float(obj.find('bndbox/xmax').text)
            ymax = float(obj.find('bndbox/ymax').text)

            x_center = (xmin + xmax) / (2 * image_width)
            y_center = (ymin + ymax) / (2 * image_height)
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            yolo_label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"Processed: {xml_file}")

print("Conversion to YOLOv5 format and data split completed.")
