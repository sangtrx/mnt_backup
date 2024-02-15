from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='hand_arm.yaml', 
            epochs=300, 
            imgsz=256, 
            batch=1, 
            name='yolov8x_hand_arm_full_aug',
            # hsv_h = 0,
            # hsv_s = 0,
            # hsv_v = 0,
            # translate = 0,
            # scale = 0,
            # fliplr = 0,
            # mosaic = 0
            )
