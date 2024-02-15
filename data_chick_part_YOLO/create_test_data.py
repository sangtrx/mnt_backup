import os
import shutil
from pathlib import Path
import chardet

def create_yolo_test_dataset(input_image_folder, input_txt_folder, filelist_path, output_dataset_folder):
    output_images_folder = os.path.join(output_dataset_folder, "images")
    output_labels_folder = os.path.join(output_dataset_folder, "labels")

    Path(output_images_folder).mkdir(parents=True, exist_ok=True)
    Path(output_labels_folder).mkdir(parents=True, exist_ok=True)

    # Detect the encoding of the file
    with open(filelist_path, "rb") as f:
        raw_data = f.read()
    encoding = chardet.detect(raw_data)["encoding"]

    # Read the file with the detected encoding
    with open(filelist_path, "r", encoding=encoding) as file:
        filenames = file.readlines()

    for filename in filenames:
        filename = filename.strip()
        # Move image file
        input_image_file = os.path.join(input_image_folder, filename)
        output_image_file = os.path.join(output_images_folder, filename)
        shutil.copy(input_image_file, output_image_file)

        # Move .txt file
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        input_txt_file = os.path.join(input_txt_folder, txt_filename)
        output_txt_file = os.path.join(output_labels_folder, txt_filename)
        shutil.copy(input_txt_file, output_txt_file)



part1_input_image_folder = "/mnt/tqsang/chicken_part1/part1_cropped_frames"
part1_input_txt_folder = "/mnt/tqsang/part1_filename/txt"
part1_filelist_path = "/mnt/tqsang/part1_filename/filelist.txt"

part2_input_image_folder = "/mnt/tqsang/chicken_part2/part2_cropped_frames"
part2_input_txt_folder = "/mnt/tqsang/part2_filename/txt"
part2_filelist_path = "/mnt/tqsang/part2_filename/filelist.txt"

output_dataset_folder = "/mnt/tqsang/data_chick_part_YOLO/datasets/test"

create_yolo_test_dataset(part1_input_image_folder, part1_input_txt_folder, part1_filelist_path, output_dataset_folder)
create_yolo_test_dataset(part2_input_image_folder, part2_input_txt_folder, part2_filelist_path, output_dataset_folder)
