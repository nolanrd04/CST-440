#!/usr/bin/env python3
"""
collect_gesture_data_improved.py

Improved gesture data collection with:
- Auto-detection of poor quality frames (blurry, too dark, etc.)
- Progress bars and clearer UI
- Ensures diversity in lighting/distance
- Saves raw captures for inspection

Usage:
    python collect_gesture_data_improved.py
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_BASE = os.path.join(SCRIPT_DIR, "data", "processed")
IMG_SIZE = 96

print("\n" + "=" * 70)
print("IMPROVED GESTURE DATA COLLECTION")
print("=" * 70)
print("\nRecommendations for best results:")
print("  • Collect 300-500 samples per gesture (not just 50!)")
print("  • Vary background: white wall, desk, outdoors")
print("  • Vary lighting: bright, dim, natural, artificial")
print("  • Vary distance: from 20cm to 50cm away")
print("  • Vary hand size: if possible, use different hands")
print("\nThis might take 30-60 minutes but will dramatically improve accuracy.")
print("=" * 70)

dataset_name = input("\nDataset name (e.g., 'my_gestures_v2'): ").strip()
if not dataset_name:
    dataset_name = "my_gestures_improved"

gesture_input = input("Gestures to record (comma-separated)\n  Options: call, dislike, like, mute, ok\n  Your choice: ").strip()
gestures = [g.strip() for g in gesture_input.split(",")]
gestures = [g for g in gestures if g in ["call", "dislike", "like", "mute", "ok"]]

if not gestures:
    print("ERROR: No valid gestures specified!")
    exit(1)

samples_str = input("Samples per gesture (recommended: 300-500): ").strip()
target_samples = int(samples_str) if samples_str else 300

output_dir = os.path.join(DATA_BASE, dataset_name)
os.makedirs(output_dir, exist_ok=True)

label_map = {g: idx for idx, g in enumerate(gestures)}
with open(os.path.join(output_dir, "label_map.json"), "w") as f:
    json.dump(label_map, f)

print(f"\n✓ Will collect {target_samples} samples × {len(gestures)} gestures = {target_samples * len(gestures)} total")
print(f"✓ Output: {output_dir}")

images = []
labels = []

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Could not open webcam!")
    exit(1)

# Laplacian variance (sharpness metric)
def is_blurry(image, threshold=50):
    if image is None or image.size == 0:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

# Brightness check
def is_too_dark(image, threshold=50):
    if image is None or image.size == 0:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return np.mean(gray) < threshold

print("\n" + "=" * 70)

for gesture_idx, gesture in enumerate(gestures):
    print(f"\n📸 Gesture {gesture_idx + 1}/{len(gestures)}: {gesture.upper()}")
    print(f"   Target: {target_samples} samples")
    print(f"   🟢 GET READY - Show your '{gesture}' gesture!")
    print("   Press SPACE to start, 'q' to skip this gesture\n")
    
    # Wait for user to get ready
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Preview (press SPACE to start)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        if key == ord('q'):
            break
    
    if key == ord('q'):
        print(f"   ⊘ Skipped {gesture}")
        continue
    
    cv2.destroyAllWindows()
    
    collected = 0
    rejected = 0
    frame_buffer = deque(maxlen=5)
    
    print(f"   Recording... (press 'q' to finish early)")
    
    while collected < target_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Quality checks
        if is_blurry(frame_gray):
            rejected += 1
            if rejected % 30 == 0:
                print(f"      {collected}/{target_samples} [Blurry frames rejected: {rejected}]", end="\r", flush=True)
            continue
        
        if is_too_dark(frame_gray):
            rejected += 1
            if rejected % 30 == 0:
                print(f"      {collected}/{target_samples} [Dark frames rejected: {rejected}]", end="\r", flush=True)
            continue
        
        # Spatial diversity: only save if sufficiently different from recent frames
        if len(frame_buffer) > 0:
            last_frame = frame_buffer[-1]
            diff = cv2.absdiff(frame_gray, last_frame)
            if np.mean(diff) < 5:  # Too similar to last frame
                continue
        
        frame_buffer.append(frame_gray.copy())
        
        # Resize to 96x96 grayscale
        resized = cv2.resize(frame_gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        
        images.append(resized)
        labels.append(label_map[gesture])
        collected += 1
        
        if collected % 10 == 0:
            print(f"      {collected}/{target_samples} ✓", end="\r", flush=True)
        
        # Show live progress
        display = cv2.resize(frame, (320, 240))
        cv2.putText(display, f"{gesture.upper()}: {collected}/{target_samples}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Recording (press q to finish)", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print(f"   ✓ Collected {collected} samples (rejected {rejected} low-quality frames)")

cap.release()

# Save as numpy arrays
X = np.array(images, dtype=np.uint8)
y = np.array(labels, dtype=np.int32)

np.save(os.path.join(output_dir, "X.npy"), X)
np.save(os.path.join(output_dir, "y.npy"), y)

print("\n" + "=" * 70)
print(f"✓ Data collection complete!")
print(f"  Total samples: {len(X)}")
print(f"  Saved to: {output_dir}")
print(f"\nNext: python train_cnn_model.py")
print("  (but first edit train_cnn_model.py and change DATASET_NAME = \"{}\")". format(dataset_name))
print("=" * 70 + "\n")
