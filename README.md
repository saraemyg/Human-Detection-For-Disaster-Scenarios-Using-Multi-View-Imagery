Human Detection in Disaster Scenarios Using Multi-View Aerial Imagery

>> Overview :

This project presents a real-time human detection pipeline designed for emergency response scenarios using aerial drone footage. It uses YOLOv8 for detection and OpenCV-based multi-view analysis to improve spatial awareness using videos from different viewpoints (e.g., closer and aerial angles). The goal is to support search-and-rescue missions by accurately identifying humans from challenging aerial perspectives in real-time, even under occlusion or cluttered environments.

>> Requirements : 
- Python 3.8+
- OpenCV (`opencv-python`)
- Ultralytics YOLO (`ultralytics`)
- NumPy
- Pygame (for alert sound)
- Scikit-learn
- Pandas




>> How to Use :
1. Run Real-Time Human Detection
Detects humans in video1.mp4 and logs results:

python human_detection_pipeline.py

2. Evaluate Against Ground Truth
Compares predictions to groundtruth.csv and prints precision, recall, IoU, etc.:

python evaluation.py
s
3. Multi-View Visualization
Aligns aerial video with original view and projects detections:

python multi_view_match.py

4. Ground Truth Generator
This is for generating groundtruth.csv primarily from Yolov8 to ease ground truth creation process

python gt_generator.py




>> Description of Python Scripts : 
human_detection_pipeline.py
- Runs YOLOv8 on each frame.
- Applies motion filtering and polygon masking to reduce false alarms.
- Logs bounding boxes to detections.csv.
- Plays sound alert if human dissappears.
- Records video segments upon detection.

evaluation.py
- Loads both prediction and ground truth CSV files.
- Computes Precision, Recall, F1-Score, FPR, FNR, and Avg IoU.
- Reports a detection error score and prints a full evaluation report.

multi_view_match.py
- Matches keypoints between original and aerial views using ORB features.
- Applies homography to project human positions across views.
- Draws green boxes on detected humans and visual correspondence lines.