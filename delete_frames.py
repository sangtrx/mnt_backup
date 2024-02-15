import os

txt_file_path = "/mnt/tqsang/delete_frames.txt"
output_deleted_files_txt = "deleted_files.txt"

root_folder = "/mnt/tqsang"
part1_xml_folder = os.path.join(root_folder, "chicken_part1", "xml_44606")
part2_xml_folder = os.path.join(root_folder, "chicken_part2", "xml1_4893")

with open(txt_file_path, "r") as input_file, open(output_deleted_files_txt, "w") as output_file:
    for line in input_file:
        line = line.strip()
        
        # Extract part number and frame number from the file path
        part_number = int(line.split("/")[-1].split("_")[0][-1])
        frame_number = int(line.split("/")[-1].split("_")[1])

        if part_number == 1:
            xml_folder = part1_xml_folder
        else:
            xml_folder = part2_xml_folder

        xml_file_path = os.path.join(xml_folder, f"{frame_number:08d}.xml")

        if os.path.exists(xml_file_path):
            os.remove(xml_file_path)
            output_file.write(f"{xml_file_path}\n")
            print(f"Deleted: {xml_file_path}")
        else:
            print(f"Skipped: {xml_file_path}")
