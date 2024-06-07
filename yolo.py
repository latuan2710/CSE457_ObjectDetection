import cv2
from ultralytics import YOLO

# Build a YOLOv9c model from pretrained weight
model = YOLO('materials/yolov9c.pt')

# Display model information
model.info()

# Detect videos
# results = model.predict("videos/london-walk-from-oxford-street-to-carnaby-street.mp4", show=True)

# Detect By Webcam
# results = model.predict(0, show=True)

# Detect images
results = model(["images/zoo.png", "images/dog.png", "images/doggo.jpg"])
for result in results:
    cv2.imshow("Yolo Detect", cv2.resize(result.plot(), (700, 500)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
