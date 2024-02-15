import os
import shutil
from pathlib import Path
import chardet
from PIL import Image

def create_test_dataset(input_folder, txt_folder, output_folder, filelist_path):
    img_output_folder = os.path.join(output_folder, "images")
    label_output_folder = os.path.join(output_folder, "labels")
    Path(img_output_folder).mkdir(parents=True, exist_ok=True)
    Path(label_output_folder).mkdir(parents=True, exist_ok=True)

    with open(filelist_path, "rb") as f:
        result = chardet.detect(f.read())

    with open(filelist_path, "r", encoding=result["encoding"]) as f:
        for line in f:
            img_name = line.strip()
            txt_name = img_name.replace(".png", ".txt")

            img_path = os.path.join(input_folder, img_name)
            txt_path = os.path.join(txt_folder, "txt", txt_name)

            if os.path.exists(img_path) and os.path.exists(txt_path):
                img = Image.open(img_path)
                width, height = img.size

                with open(txt_path, "r") as fr:
                    with open(os.path.join(label_output_folder, txt_name), "w") as fw:
                        for box_line in fr:
                            cls, x, y, w, h = map(float, box_line.strip().split())
                            x = x / width
                            y = y / height
                            w = w / width
                            h = h / height
                            fw.write(f"{int(cls)} {x} {y} {w} {h}\n")

                shutil.copy(img_path, os.path.join(img_output_folder, img_name))


base_path = "/mnt/tqsang/data_chick_part_YOLO/datasets"

input_folder_part1 = "/mnt/tqsang/chicken_part1/part1_cropped_frames"
# txt_folder_part1 = "/mnt/tqsang/part1_filename"
txt_folder_part1 = "/mnt/tqsang/data_chick_part_YOLO/part1"

filelist_part1 = "/mnt/tqsang/part1_filename/filelist.txt"

input_folder_part2 = "/mnt/tqsang/chicken_part2/part2_cropped_frames"
# txt_folder_part2 = "/mnt/tqsang/part2_filename"
txt_folder_part2 = "/mnt/tqsang/data_chick_part_YOLO/part2"

filelist_part2 = "/mnt/tqsang/part2_filename/filelist.txt"

output_folder = os.path.join(base_path, "test")

create_test_dataset(input_folder_part1, txt_folder_part1, output_folder, filelist_part1)
create_test_dataset(input_folder_part2, txt_folder_part2, output_folder, filelist_part2)
