from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='skin.yaml', 
            epochs=1000, 
            imgsz= (512,512), 
            patience = 100,
            batch=16, 
            name='yolov8x_skin_dot3_noMosaic_add',
            mosaic = 0,   # turn this off 
            scale = 0.2  # turn this off               
            )
