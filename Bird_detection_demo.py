from ultralytics import YOLO
import os
import cv2

yaml_file = "demo_dataset.yaml"
image_dir = r"D:\\deep learning\\demo_dataset\\images"

model = YOLO("yolov8n.pt")
model.train(data="demo_dataset.yaml", epochs=2, imgsz=416, batch=2)

# Inference on an image
img_files = os.listdir(image_dir)
img_path = os.path.join(image_dir, img_files[0])
img = cv2.imread(img_path)
results = model(img)
cv2.imshow("Bird Detection Demo", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
