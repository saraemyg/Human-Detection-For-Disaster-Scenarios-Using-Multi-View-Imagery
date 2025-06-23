# For preparing groundtruth, we generate first with the help of YOLOv8m
# Then, each images are re-evaluate to ensure the correct and accurate groundtruth to use

import os
import cv2
import pandas as pd
from ultralytics import YOLO

# === Configuration ===
GT_IMAGE_FOLDER = "gt/gt_images"
GT_CSV_PATH = "groundtruth.csv"
MODEL_PATH = "yolov8m.pt"  # Use a detection model
CONF_THRESHOLD = 0.4
SAVE_LIMIT = 500  # Only save first 500 frames

# === Load model ===
model = YOLO(MODEL_PATH)

# === Prepare output ===
with open(GT_CSV_PATH, "w") as f:
    f.write("frame,x1,y1,x2,y2\n")

# === Process each image ===
image_files = sorted([f for f in os.listdir(GT_IMAGE_FOLDER) if f.endswith(('.jpg', '.png'))])[:SAVE_LIMIT]

for idx, image_file in enumerate(image_files):
    image_path = os.path.join(GT_IMAGE_FOLDER, image_file)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"[Warning] Skipped unreadable image: {image_path}")
        continue

    results = model.predict(frame, conf=CONF_THRESHOLD, iou=0.45)
    boxes = results[0].boxes

    for box in boxes:
        if int(box.cls[0]) == 0:  # Person class only
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            with open(GT_CSV_PATH, "a") as f:
                f.write(f"{image_file},{x1},{y1},{x2},{y2}\n")

    print(f"[{idx+1}/{len(image_files)}] Processed: {image_file}")

print("\n✅ Ground truth generation completed. Saved to:", GT_CSV_PATH)
