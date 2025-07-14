from ultralytics import YOLO

# Use absolute path to the model
model = YOLO("road_signs.pt")

# Print index-to-class name mapping
print(model.names)
