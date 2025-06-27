# Build a new model from YAML and start training from scratch
yolo detect train data=corn.yaml model=yolo11n.yaml epochs=100 imgsz=640

# Start training from a pretrained *.pt model
yolo detect train data=corn.yaml model=yolo11n.pt epochs=100 imgsz=640