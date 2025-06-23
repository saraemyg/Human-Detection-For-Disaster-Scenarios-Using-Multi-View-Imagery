# Import necessary libraries
import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
import logging
import pygame
import subprocess
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to calculate Intersection over Union (IoU) between two bounding boxes
def calculate_iou(box1, box2):
    # Determine the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    # Compute the area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    # Compute the area of each box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # Compute the union area
    union_area = box1_area + box2_area - intersection_area
    # Return IoU value (intersection over union)
    return intersection_area / union_area if union_area > 0 else 0

# Function to load bounding box data from CSV file
def load_boxes(filepath, is_gt=True):
    # Read CSV file into DataFrame
    df = pd.read_csv(filepath)
    # Check for required columns in detections file
    if not is_gt:
        if 'frame' not in df.columns:
            raise ValueError("Expected 'frame' column in detections CSV.")
        # Convert all coordinates to float
        df["x1"] = df["x1"].astype(float)
        df["y1"] = df["y1"].astype(float)
        df["x2"] = df["x2"].astype(float)
        df["y2"] = df["y2"].astype(float)
    else:
        # Same float conversion for ground truth
        df["x1"] = df["x1"].astype(float)
        df["y1"] = df["y1"].astype(float)
        df["x2"] = df["x2"].astype(float)
        df["y2"] = df["y2"].astype(float)
    # Return DataFrame with required columns
    return df[['frame', 'x1', 'y1', 'x2', 'y2']].copy()

# Function to evaluate predicted vs ground truth bounding boxes
def evaluate(gt_csv, det_csv, iou_threshold=0.5):
    # Load ground truth and detection boxes
    gt_df = load_boxes(gt_csv, is_gt=True)
    det_df = load_boxes(det_csv, is_gt=False)
    # Get union of all frame names
    all_frames = sorted(set(gt_df['frame']) | set(det_df['frame']))
    # Lists for binary classification and IoU scores
    y_true, y_pred = [], []
    iou_scores = []

    # Loop through each frame to compare predictions and ground truth
    for frame in all_frames:
        gt_boxes = gt_df[gt_df['frame'] == frame][['x1', 'y1', 'x2', 'y2']].values
        det_boxes = det_df[det_df['frame'] == frame][['x1', 'y1', 'x2', 'y2']].values

        matched_gt = set() # Track ground truths that have already been matched
        for det_box in det_boxes:
            matched = False
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                # Compute IoU between this detection and this ground truth box
                iou = calculate_iou(det_box, gt_box)
                if iou >= iou_threshold:
                    # Match found
                    matched_gt.add(i)
                    y_true.append(1) # True positive
                    y_pred.append(1)
                    iou_scores.append(iou)
                    matched = True
                    break
                else:
                    # Print low IoU matches for debugging
                    print(f"Low IoU ({iou:.3f}) - Frame: {frame}, Detection: {det_box}, GT: {gt_box}")
            if not matched:
                y_true.append(0) # False positive
                y_pred.append(1)
        # Count missed detections (false negatives)
        for i in range(len(gt_boxes)):
            if i not in matched_gt:
                y_true.append(1) # False negative
                y_pred.append(0)

    # Compute evaluation metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    error = 1 - precision
    total_pred = y_pred.count(1)
    total_gt = y_true.count(1)
    fpr = fp / max(1, (fp + total_gt))  # False positive rate
    fnr = fn / max(1, (fn + total_gt))  # False negative rate

    # Print summary
    print("\n--- Evaluation Report ---")
    print(f"Precision: {precision:.3f}")
    print(f"Error:     {error:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")
    print(f"FPR:       {fpr:.3f}")
    print(f"FNR:       {fnr:.3f}")
    print(f"Avg IoU:   {np.mean(iou_scores):.3f}" if iou_scores else "Avg IoU: N/A (no matches)")

# Main function to run evaluation on CSV files
if __name__ == '__main__':
    evaluate(gt_csv='groundtruth.csv', det_csv='detections.csv')
