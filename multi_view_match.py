import cv2
import numpy as np
from ultralytics import YOLO

# Load the original ground-level video (closer angle)
video_original = cv2.VideoCapture('video/video3.mp4')
# Load the aerial video (higher altitude, wider view)
video_aerial = cv2.VideoCapture('video/video4.mp4') 

# This model will be used to detect humans only in the original view
model = YOLO('yolov8m.pt')  

# ORB is used to extract keypoints and descriptors for image matching
orb = cv2.ORB_create(3000) # Detect up to 3000 keypoints
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # Use Hamming distance and enable cross-check

# Begin frame-by-frame processing loop
while True:
    # Read the next frame from both videos
    ret1, frame_orig = video_original.read()
    ret2, frame_aerial = video_aerial.read()
    # If either video ends, stop the loop
    if not ret1 or not ret2:
        break
    # Resize both frames to a manageable display size
    height = 400 #450
    width = 750 #800
    frame_orig = cv2.resize(frame_orig, (width, height))
    frame_aerial = cv2.resize(frame_aerial, (width, height))
    # Human Detection with YOLOv8 (on original frame only)
    results = model.predict(frame_orig, conf=0.4) # Set confidence threshold
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
    classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
    # Filter out only the human detections (COCO class 0 = person)
    human_boxes = []
    for box, cls in zip(boxes, classes):
        if int(cls) == 0:  # Class 0 = human/person in COCO
            x1, y1, x2, y2 = map(int, box)
            human_boxes.append((x1, y1, x2, y2))

    # Convert both frames to grayscale for keypoint detection
    gray_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
    gray_aerial = cv2.cvtColor(frame_aerial, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors for each frame
    kp1, des1 = orb.detectAndCompute(gray_orig, None)
    kp2, des2 = orb.detectAndCompute(gray_aerial, None)

    # If descriptors not found in either image, skip this frame
    if des1 is None or des2 is None:
        continue
    # Match Features: If descriptors not found in either image, skip this frame
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance) # Sort matches by quality (lower distance = better match)

    # Use top-N matches for homography estimation 
    good_matches = matches[:50] # Keep only the top 50 matches for estimating transformation
    if len(good_matches) < 10:
        continue # Not enough matches for reliable homography

    # Extract matched keypoint coordinates from both images
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) # From original view
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) # From aerial view

    # Estimate homography matrix to align aerial image to original
    H, mask = cv2.findHomography(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    # Warp aerial frame to match perspective of original frame
    aligned_aerial = cv2.warpPerspective(frame_aerial, H, (width, height))

    # Visualize keypoint matches (gray lines)
    # Stack both images vertically: original on top, aerial at bottom
    match_vis = np.vstack((frame_orig.copy(), frame_aerial.copy()))
    for m in good_matches:
        pt1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))  # From original
        pt2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int))  # From aerial
        pt2_shifted = (pt2[0], pt2[1] + height)  # shift aerial keypoints downward
        cv2.line(match_vis, pt1, pt2_shifted, (180, 180, 180), 1) # Draw gray line

    # Draw green boxes around humans and draw green lines projecting to corresponding aerial points
    for (x1, y1, x2, y2) in human_boxes:
        # Draw bounding box on original frame (top)
        cv2.rectangle(match_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Compute center point of the box and project to aerial frame
        center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype='float32').reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(center, np.linalg.inv(H)) # Map to aerial perspective
        px, py = projected[0][0]
        # Draw projection line (green) from original to aerial view
        cv2.line(match_vis, (int((x1 + x2) / 2), int((y1 + y2) / 2)), (int(px), int(py) + height), (0, 255, 0), 2)

    # Add labels for clarity to both views
    cv2.putText(match_vis, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(match_vis, "Aerial", (10, height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the visualization window
    cv2.imshow("Original (Top) vs Aerial (Bottom)", match_vis)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup resources after video ends or user exits
video_original.release()
video_aerial.release()
cv2.destroyAllWindows()
