from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='hand.yaml', 
            epochs=500, 
            imgsz=256, 
            batch=1, 
            name='yolov8x_hand_full_aug_v2_256_bs1',
            flipud = 0.1, #v2 
            hsv_h = 0.2, # v2 0.2 #v3 0.5
            # hsv_s = 0,
            # hsv_v = 0,
            # translate = 0,
            # scale = 0,
            # fliplr = 0,
            # mosaic = 0
            )
