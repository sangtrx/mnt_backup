from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='scale.yaml', 
            epochs=500, 
            imgsz=256, 
            batch=1, 
            name='yolov8x_scale_v2_ShiftScaleRotate',
            flipud = 0.5, #v2 
            hsv_h = 0.2, # v2 0.2 #v3 0.5
            # hsv_s = 0,
            # hsv_v = 0,
            # translate = 0,
            # scale = 0,
            # fliplr = 0,
            # mosaic = 0
            )
