from ultralytics import YOLO

# Load a model
model = YOLO('/mnt/tqsang/hand_dataset_YOLO/runs/detect/yolov8x_hand_full_aug_v3_256_bs1/weights/best.pt')  # load a pretrained model (recommended for training)

# Validate the model
metrics = model.val(data='/mnt/tqsang/hand_dataset_YOLO/hand.yaml',split='train')  # no arguments needed, dataset and settings remembered
metrics.confusion_matrix
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category