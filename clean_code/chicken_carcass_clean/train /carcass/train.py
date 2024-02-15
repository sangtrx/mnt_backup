from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='carcass.yaml', 
            epochs=1000, 
            patience = 100,
            imgsz=256, 
            batch=64, 
            name='yolov8x_brio_det_only',
            hsv_h = 0.01, # v2
            hsv_s = 0.05,
            hsv_v = 0.05,
            translate = 0,
            scale = 0.05,
            fliplr = 0,
            mosaic = 0            
            )
