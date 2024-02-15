from ultralytics import YOLO

# part_config = {
#     'feather': {'model_path': '/home/tqsang/JSON2YOLO/feather_YOLO_det/runs/detect/yolov8x_dot3_noRot/weights/best.pt', 'conf_threshold': 0.15},
#     'feather on skin': {'model_path': '/home/tqsang/JSON2YOLO/feather_on_skin_YOLO_det/runs/detect/yolov8x_feather_on_skin/weights/best.pt', 'conf_threshold': 0.15},
#     'wing': {'model_path': '/home/tqsang/JSON2YOLO/wing_YOLO_det/runs/detect/yolov8x_wing_dot3_nohardcase_noMosaic_noScale/weights/best.pt', 'conf_threshold': 0.2},
#     'skin': {'model_path': '/home/tqsang/JSON2YOLO/skin_YOLO_det/runs/detect/yolov8x_skin_dot3_noMosaic_add/weights/best.pt', 'conf_threshold': 0.2},
#     'flesh': {'model_path': '/home/tqsang/JSON2YOLO/flesh_YOLO_det/runs/detect/yolov8x_flesh_dot3/weights/best.pt', 'conf_threshold': 0.4}
# }

# # loop over models and export
# for part_name, config in part_config.items():
#     # Load models
#     model = YOLO(model=config['model_path'])  # load a specific model
    
#     # Export the model
#     model.export(format='engine', imgsz = 512, device=1)


model = YOLO(model='/home/tqsang/JSON2YOLO/carcass_brio_YOLO_det_only/runs/detect/yolov8x_brio_det_only/weights/best.pt')  # load a specific model

# Export the model
model.export(format='engine', imgsz = 256, device=1)