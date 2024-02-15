from ultralytics import YOLO

# Load a model
model = YOLO('/mnt/tqsang/data_chick_part_YOLO/runs/detect/yolov8x_default_rot/weights/best.pt')  # load a pretrained model (recommended for training)

# Validate the model
metrics = model.val(data='chick_part.yaml',split='val')  # no arguments needed, dataset and settings remembered
metrics.confusion_matrix
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category