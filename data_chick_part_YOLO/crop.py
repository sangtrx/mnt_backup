from pathlib import Path
import os
import chardet
from PIL import Image

def crop_remaining_frames(filelist_path, img_folder, output_folder, areas):
    Path(output_folder).mkdir(exist_ok=True)

    # Detect the encoding of the file
    with open(filelist_path, "rb") as f:
        raw_data = f.read()
    encoding = chardet.detect(raw_data)["encoding"]

    # Read the file with the detected encoding
    with open(filelist_path, "r", encoding=encoding) as file:
        filenames = file.readlines()

    for filename in filenames:
        filename = filename.strip()
        frame_number = int(filename.split('_')[0])

        img_file = f"{frame_number:08d}.png"
        img = Image.open(os.path.join(img_folder, img_file))

        for i, area in enumerate(areas):
            cropped = img.crop(area)
            cropped_name = f"{frame_number:08d}_{'L' if i == 0 else 'R'}.png"
            cropped.save(os.path.join(output_folder, cropped_name))

root_folder = "/mnt/tqsang"
part1_folder = os.path.join(root_folder, "chicken_part1")

part1_img_folder = os.path.join(part1_folder, "frames")
areas_part1 = [(240, 100, 1008, 868), (878, 139, 1646, 907)]

output_folder_part1 = "/mnt/tqsang/chicken_part1/part1_cropped_frames"

part1_filelist_path = "/mnt/tqsang/part1_filename/filelist.txt"

crop_remaining_frames(part1_filelist_path, part1_img_folder, output_folder_part1, areas_part1)
