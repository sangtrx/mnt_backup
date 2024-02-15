import os
import json
import numpy as np

in_path = '/media/Seagate16T/tqsang/vidcap/data/aKhoa'
out_path = '/media/Seagate16T/tqsang/vidcap/data/out'

# if not os.path.exists(out_path):
#     os.mkdir(out_path)

# os.chdir(in_path) # change directory from working dir to dir with files
extension = '.json'
for item in os.listdir(in_path): # loop through items in dir
    if item.endswith(extension): # check for ".zip" extension
        file_name = os.path.abspath(item) # get full path of files
        print(item.replace(".json", ""))

        key = item.replace(".json", "")

        with open(in_path+'/'+item ) as json_file:
            data = json.load(json_file)

        sav_data = np.array([])
        for i in range(100):
          sav_data = np.append(sav_data , np.array(data['video_features'][i]['features'][0], dtype=np.float32))

        sav_data = np.reshape(sav_data, (100, 2048))
        np.save(os.path.join(out_path, key+'.npy'), sav_data)
