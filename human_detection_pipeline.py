# Import core libraries
import os
import cv2
import time
import numpy as np
from ultralytics import YOLO  # YOLOv8 model for object detection
import logging
import pygame  # For playing sound alerts
import subprocess  # For running the evaluation script after execution

class EnhancedHumanRecorder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.setup_logging()
        self.model = YOLO(cfg['model_path'])  # Load YOLOv8 model
        self.cap = cv2.VideoCapture(cfg['Video In'])  # Open input video
        # Initialize background subtraction for motion filtering
        self.motion = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50) 
        self.ignore_mask = self.create_ignore_mask()  # Optional polygon mask to ignore unwanted regions
        # Internal flags and states
        self.writer = None
        self.recording = False
        self.frame_id = 0
        self.prev_detected = False
        self.last_seen_frame = -1
        self.last_no_detect_time = time.time()
        # Calculate frame rate considering skip settings
        self.adjusted_fps = self.cap.get(cv2.CAP_PROP_FPS) / cfg['FRAME_SKIP'] if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 15
        self.last_boxes = []
        self.last_boxes_frame = -1
        # Initialize audio for alert notification
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound("video/sound/sound_small.mp3")
        # Buffer to keep past detections in memory for a few seconds
        self.memory_buffer = int(self.adjusted_fps * cfg.get('MEMORY_SECONDS', 2))
        self.logged_frames = 0
        # Prepare CSV for logging detection bounding boxes
        self.detection_log_path = "detections.csv"
        if os.path.exists(self.detection_log_path):
            os.remove(self.detection_log_path)
        with open(self.detection_log_path, "w") as f:
            f.write("frame,x1,y1,x2,y2\n")
        # Create folder for recordings
        os.makedirs('Rec', exist_ok=True)

    def setup_logging(self): #logging setup
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        self.logger = logging.getLogger('Recorder')

    def create_ignore_mask(self): #polygon ignore mask
        mask = np.ones((self.cfg['RESIZE_HEIGHT'], self.cfg['RESIZE_WIDTH']), dtype=np.uint8) * 255
        polygon = np.array([[0, 420], [960, 420], [960, 540], [0, 540]])
        cv2.fillPoly(mask, [polygon], 0) # Mask out bottom static region (e.g., vehicles, grass)
        return mask

    def run(self): #main loop
        while self.cap.isOpened():
            ret, original_frame = self.cap.read()
            if not ret:
                break

            if self.frame_id % self.cfg['FRAME_SKIP'] != 0:
                self.frame_id += 1
                continue
            # Resize and blur for motion detection
            resized_frame = cv2.resize(original_frame, (self.cfg['RESIZE_WIDTH'], self.cfg['RESIZE_HEIGHT']))
            frame_blur = cv2.GaussianBlur(resized_frame, (3, 3), 0)
            # Apply motion detection and region filtering
            motion_raw = self.motion.apply(frame_blur)
            motion_mask = cv2.bitwise_and(motion_raw, self.ignore_mask)
            motion_roi = cv2.bitwise_and(resized_frame, resized_frame, mask=motion_mask)
            # Run YOLOv8 on the motion region
            results = self.model.predict(motion_roi, conf=self.cfg['CONF'], iou=0.45)
            boxes = results[0].boxes
            masks = results[0].masks.data if results[0].masks is not None else None

            detected = False
            current_boxes = []
            # Scale factors for converting bbox to original frame size
            scale_x = original_frame.shape[1] / self.cfg['RESIZE_WIDTH']
            scale_y = original_frame.shape[0] / self.cfg['RESIZE_HEIGHT']

            # Detction Filtering and Scaling
            for i, box in enumerate(boxes):
                if int(box.cls[0]) == 0: # Class 0 = human
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
 
                    # Reject weak segmentation masks or masks with too little area
                    if masks is not None:
                        mask = masks[i]
                        mask_area = mask.sum()
                        box_area = (x2 - x1) * (y2 - y1)
                        if mask_area < 100 or mask_area / box_area < 0.25:
                            continue
                    # Reject small boxes (noise or irrelevant detections)
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area < 800 or (y2 - y1) < 40:
                        continue
                    # Slight padding
                    pad_x = int((x2 - x1) * 0.05)
                    pad_y = int((y2 - y1) * 0.05)
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(self.cfg['RESIZE_WIDTH'] - 1, x2 + pad_x)
                    y2 = min(self.cfg['RESIZE_HEIGHT'] - 1, y2 + pad_y)
                    # Scale back to original frame size
                    x1_scaled = int(x1 * scale_x)
                    y1_scaled = int(y1 * scale_y)
                    x2_scaled = int(x2 * scale_x)
                    y2_scaled = int(y2 * scale_y)

                    current_boxes.append(((x1_scaled, y1_scaled, x2_scaled, y2_scaled), conf))
                    detected = True
            # timestampt overlay
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(resized_frame, timestamp, (10, self.cfg['RESIZE_HEIGHT'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # logging and alerting
            if detected:
                self.last_boxes = current_boxes
                self.last_boxes_frame = self.frame_id
                self.last_seen_frame = self.frame_id
                self.last_no_detect_time = time.time()
                for (x1, y1, x2, y2), conf in current_boxes:
                    if self.logged_frames < 500:
                        with open(self.detection_log_path, "a") as f:
                            f.write(f"frame_{self.frame_id:04d}.jpg,{x1},{y1},{x2},{y2}\n")
                        self.logged_frames += 1

                    cv2.rectangle(resized_frame, (int(x1 / scale_x), int(y1 / scale_y)),
                                  (int(x2 / scale_x), int(y2 / scale_y)), (0, 255, 0), 2)
                    cv2.putText(resized_frame, f"Human ({conf:.2f})", (int(x1 / scale_x), int(y1 / scale_y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if not self.prev_detected:
                    self.logger.info(f"[ALERT] Human REAPPEARED at frame {self.frame_id}")
                if not self.recording:
                    self.start_recording()
            # memory buffer for short-term tracking
            else:
                frames_since_last_seen = self.frame_id - self.last_boxes_frame

                if frames_since_last_seen <= self.memory_buffer:
                    for (x1, y1, x2, y2), conf in self.last_boxes:
                        cv2.rectangle(resized_frame, (int(x1 / scale_x), int(y1 / scale_y)),
                                      (int(x2 / scale_x), int(y2 / scale_y)), (0, 255, 255), 2)
                        cv2.putText(resized_frame, f"Last Seen ({conf:.2f})", (int(x1 / scale_x), int(y1 / scale_y) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(resized_frame, "Human is recently present", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                else:
                    if self.prev_detected:
                        self.logger.info(f"[ALERT] Human DISAPPEARED at frame {self.frame_id}")
                    if self.recording:
                        self.stop_recording()
                    cv2.putText(resized_frame, "No Detection", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    self.alert_sound.play()
            # visualisation and controls
            self.prev_detected = detected
            cv2.imshow('Human Detection', resized_frame)
            if self.recording:
                self.writer.write(resized_frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            self.frame_id += 1

        self.cleanup()

    # Helpers
    def start_recording(self):
        filename = os.path.join('Rec', time.strftime("%Y-%m-%d_%H-%M-%S") + ".avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(filename, fourcc, self.adjusted_fps,
                                      (self.cfg['RESIZE_WIDTH'], self.cfg['RESIZE_HEIGHT']))
        self.recording = True
        self.logger.info(f"Recording started: {filename}")

    def stop_recording(self):
        self.recording = False
        if self.writer:
            self.writer.release()
        self.logger.info("Recording stopped")

    # Cleanup resources after video ends or user exits
    def cleanup(self):
        self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

# configuration & launch
def parse_cfg():
    return {
        'Video In': 'video/video1.mp4',
        'FRAME_SKIP': 1,
        'RESIZE_WIDTH': 960,
        'RESIZE_HEIGHT': 540,
        'CONF': 0.40,
        'model_path': 'yolov8m.pt',
        'MOTION_THRESHOLD': 500,
        'MEMORY_SECONDS': 2
    }

if __name__ == '__main__':
    cfg = parse_cfg()
    rec = EnhancedHumanRecorder(cfg)
    rec.run()
    subprocess.run(['python', 'evaluation.py']) # Automatically evaluate results after run
