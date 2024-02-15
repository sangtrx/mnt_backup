import os
import cv2
import json
import time
import torch
import numpy as np
# from custom_inference import grcnn_get_obj_bboxes
# from lib.scene_parser.rcnn.structures.image_list import to_image_list
# 
# from custom_inference import grcnn_get_obj_bboxes
# from torch.utils.data.dataloader import DataLoader
# from torch.utils.data.dataset import Dataset
# from lib.data.build import build_data_loader
from tqdm import tqdm


# class ANetData(Dataset):
#     def __init__(self, frames):
#         self.frames = frames
# 
#     def __len__(self):
#         return len(self.frames)
# 
#     def __getitem__(self, index):
#         return self.frames[index]

def my_visualize_func(img, bboxes, class_names, image_name):
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
        cv2.putText(img, class_names[i], (int(x1), int(y1)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imwrite("visualize/result" + image_name + ".jpg", img)

def get_json_format(info):
    frame_bboxes = []
    for one_box_info in info:
        frame_bboxes.append({
            "box": str(one_box_info[0]), 
            "score": str(one_box_info[1]), 
            "class_id": str(one_box_info[2])
        })

    one_frame_info = {
        "pts": 0,
        "frame_bboxes": frame_bboxes
    }

    return one_frame_info

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--div', type=int, default=1)
    args = parser.parse_args()
    vid_dir = '/media/Seagate16T/datasets/ActivityNet1.3/anet_rescaled/anet_videos_rescaled_1600'
    frame_dir = '/media/Seagate16T/tqminh/ActNetFrames/'
    try:
        os.makedirs(frame_dir)
    except:
        pass
    bbox_json_dir = '/media/Seagate16T/tqminh/bbox_obj_json'
    batch_size = 8
    vid_names = sorted(os.listdir(vid_dir))
    '''
    n = len(vid_names)
    vid_names = vid_names[1961: n//4]
    if args.div == 1:
        start_indx = 0
        end_indx = len(vid_names)//3 - 1

    elif args.div == 2:
        start_indx = len(vid_names) // 3
        end_indx = 2 * (len(vid_names)//3) - 1
    elif args.div == 3:
        start_indx = 2 * (len(vid_names)//3)
        end_indx = len(vid_names) - 1
    '''
 
    '''
    if args.div == 1:
        start_indx = 0
        end_indx = len(vid_names)//4 - 1
        # failed_vid_name = 'v_5P-4_nS8euM.mp4'
        # for i in range(start_indx, end_indx + 1):
        #     if  vid_names[i] == failed_vid_name:
        #         failed_idx = i
        #         break
        # print(failed_idx, vid_names[failed_idx], failed_vid_name)
        # exit(0)
    elif args.div == 2:
        start_indx = len(vid_names) // 4
        end_indx = 2 * (len(vid_names)//4) - 1
    elif args.div == 3:
        start_indx = 2 * (len(vid_names)//4)
        end_indx = 3 * (len(vid_names)//4) - 1
    elif args.div == 4:
        start_indx = 3 * (len(vid_names)//4)
        end_indx =  len(vid_names) - 1

    vid_names = vid_names[start_indx: end_indx + 1]
    print('Batch {}, from {} - {}, num of videos: {}'.format(args.div, start_indx+1961, 
                                                            end_indx+1961, len(vid_names)))
    '''

    for vid_name in tqdm(vid_names):
        start = time.time()
        vid_name_path = os.path.join(vid_dir, vid_name)
        frame_folder = os.path.join(frame_dir, os.path.splitext(vid_name)[0])
        try:
            os.makedirs(frame_folder)
        except:
            pass
#self.rgb_img.save_to_disk(join(self.data_path,
                                            #'images/%07d.jpg' % self.rgb_img.frame))

        print("Video name: ", vid_name)
        vidcap = cv2.VideoCapture(vid_name_path)
        success,image = vidcap.read()
        count = 0
        middle_frames = []
        n_frames = 0
        while success:
            success,image = vidcap.read()
            count += 1
            if count % 8 == 0 and (count // 8) % 2 != 0: # maybe a middle frame of a 16-frames snipet
                n_frames += 1
                middle_frame = image# 480,720,3
                cv2.imwrite(os.path.join(frame_folder, '%d.jpg' % n_frames), middle_frame)
                middle_frame = np.moveaxis(middle_frame, -1, 0)
                middle_frames.append(middle_frame)
                
        continue

        # anet_frame_loader = DataLoader(dataset=ANetData(middle_frames), 
        #                                 batch_size=8,
        #                                 shuffle=False, 
        #                                 num_workers=64)
        # result = {
        #     "num_frames": 1600, 
        #     "secs_per_frame": 0.03333333333333333, 
        #     "video_bboxes": []
        # }

        # count_fail = 0
        # count_all = 0
        # for imgs in anet_frame_loader:
        #     imgs = imgs.to('cuda', dtype=torch.float32)
        #     imgs = to_image_list(imgs)
        #     objs_info = grcnn_get_obj_bboxes(imgs) # list of img results
        #     
        #     '''
        #     # visualize some for double checking
        #     from lib.config import cfg
        #     config_file = 'configs/sgg_res101_step.yaml'
        #     cfg.merge_from_file(config_file)
        #     cfg.resume = 0
        #     cfg.instance = -1
        #     cfg.inference = True
        #     cfg.MODEL.USE_FREQ_PRIOR = False 
        #     cfg.MODEL.ALGORITHM = 'sg_grcnn' 
        #     data_loader_test = build_data_loader(cfg, split="test", is_distributed=False)
        #     i = 4
        #     for i in range(len(imgs.tensors)):
        #         img = imgs.tensors[i]
        #         print(img.shape)
        #         vis_img = img.permute(1,2,0).cpu().numpy()
        #         # import pdb;pdb.set_trace()
        #         try:
        #             my_visualize_func(vis_img, [objs_info[i][0][0].tolist()], 
        #                             [data_loader_test.dataset.ind_to_classes[objs_info[i][0][2]]], 
        #                             'tentesi' + str(i))
        #         except:
        #             pass

        #     exit(0)
        #     '''

        #     for obj_info in objs_info:
        #         count_all += 1
        #         if obj_info == []:
        #             count_fail += 1

        #         result['video_bboxes'].append(get_json_format(obj_info))

        # dst_path = os.path.join(bbox_json_dir, os.path.splitext(vid_name)[0] + '.json')
        # with open(dst_path, 'w') as fp:
        #     json.dump(result, fp)

        # print("number of fail-to-detect frames: {}/{}".format(count_fail, count_all))
        # print('one video process time: {}s'.format(time.time() - start))
