import os
import shutil

source_folders = ['/mnt/tqsang/chicken_part1/frames', '/mnt/tqsang/chicken_part2/frames']
destination_folder = '/mnt/tqsang/scale_sample'
num_files_to_copy = 300

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Iterate through the source folders
for idx, source_folder in enumerate(source_folders):
    files = os.listdir(source_folder)
    files.sort()  # Sort files in name order
    
    # Copy the first 100 files
    for i, file in enumerate(files):
        if i >= num_files_to_copy:
            break
        
        part_name = f"part{idx+1}"
        new_file_name = f"{part_name}_{file}"
        
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, new_file_name)
        
        shutil.copy2(source_path, destination_path)
        print(f"Copied {source_path} to {destination_path}")

print("Copy process completed.")
