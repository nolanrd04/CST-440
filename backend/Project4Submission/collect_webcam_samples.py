#!/usr/bin/env python3
"""
collect_webcam_samples.py

Captures gesture samples from the webcam and appends them to an existing
dataset (or creates a new one). Uses the same center-crop → 96×96 grayscale
preprocessing as test_float32_webcam.py so the training data matches what
the model sees during inference.

Controls during recording:
    SPACE  : start / pause recording
    q      : finish current gesture, move to next
    ESC    : save and quit

Usage:
    python collect_webcam_samples.py
"""

import os
import json
import cv2
import numpy as np
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_BASE  = os.path.join(SCRIPT_DIR, "data", "processed")
IMG_SIZE   = 96

# ── Helpers ────────────────────────────────────────────────────────────────────

def center_crop_96(frame):
    """Same preprocessing as test_float32_webcam.py."""
    h, w = frame.shape[:2]
    size = min(h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    square = frame[y0:y0+size, x0:x0+size]
    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

def is_blurry(gray, threshold=50):
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def is_too_dark(gray, threshold=50):
    return np.mean(gray) < threshold

def load_existing(dataset_dir):
    """Load existing X/y arrays if present, otherwise return empty arrays."""
    x_path = os.path.join(dataset_dir, "X.npy")
    y_path = os.path.join(dataset_dir, "y.npy")
    if os.path.exists(x_path) and os.path.exists(y_path):
        X = np.load(x_path)
        y = np.load(y_path)
        print(f"  Loaded existing dataset: {len(X)} samples")
        return list(X), list(y)
    return [], []

def save_dataset(dataset_dir, images, labels):
    X = np.array(images, dtype=np.uint8)
    y = np.array(labels,  dtype=np.int32)
    np.save(os.path.join(dataset_dir, "X.npy"), X)
    np.save(os.path.join(dataset_dir, "y.npy"), y)
    return X, y

# ── UI ─────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("WEBCAM SAMPLE COLLECTOR")
print("=" * 70)
print("Captures center-cropped 96x96 frames — same preprocessing as inference.")
print("Can append to an existing dataset to mix webcam + HaGRID data.\n")

dataset_name = input("Dataset name (existing or new): ").strip()
if not dataset_name:
    print("ERROR: dataset name required"); exit(1)

dataset_dir = os.path.join(DATA_BASE, dataset_name)
label_map_path = os.path.join(dataset_dir, "label_map.json")

os.makedirs(dataset_dir, exist_ok=True)

# Load or build label map
if os.path.exists(label_map_path):
    with open(label_map_path) as f:
        label_map = json.load(f)
    print(f"  Existing label map: {label_map}")
    gesture_input = input(
        "Gestures to record (comma-separated, or Enter to use all existing): "
    ).strip()
    if gesture_input:
        new_gestures = [g.strip() for g in gesture_input.split(",") if g.strip()]
        # Add any new gestures to the label map
        for g in new_gestures:
            if g not in label_map:
                label_map[g] = max(label_map.values()) + 1
                print(f"  Added new gesture '{g}' → class {label_map[g]}")
        gestures = new_gestures
    else:
        gestures = list(label_map.keys())
else:
    gesture_input = input("Gestures to record (comma-separated): ").strip()
    gestures = [g.strip() for g in gesture_input.split(",") if g.strip()]
    if not gestures:
        print("ERROR: no gestures specified"); exit(1)
    label_map = {g: idx for idx, g in enumerate(gestures)}

with open(label_map_path, "w") as f:
    json.dump(label_map, f, indent=2)

samples_str = input("Target samples per gesture (default 300): ").strip()
target = int(samples_str) if samples_str else 300

print(f"\nGestures : {gestures}")
print(f"Target   : {target} samples each")
print(f"Output   : {dataset_dir}")
print("=" * 70)

# ── Open webcam ────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("ERROR: could not open webcam"); exit(1)

# Load existing data to append to
all_images, all_labels = load_existing(dataset_dir)

# ── Record each gesture ────────────────────────────────────────────────────────

for gesture in gestures:
    existing_count = sum(1 for lbl in all_labels if lbl == label_map[gesture])
    print(f"\nGesture: {gesture.upper()}  (already have {existing_count} samples)")
    print("  Press SPACE to start recording, ESC to save and quit\n")

    recording = False
    collected = 0
    rejected  = 0
    frame_buffer = deque(maxlen=5)
    space_was_down = False   # debounce — only toggle on fresh press

    while collected < target:
        ret, frame = cap.read()
        if not ret:
            continue

        crop = center_crop_96(frame)

        # Build display: webcam feed + small preview of what model sees
        display = frame.copy()
        h, w = display.shape[:2]
        size = min(h, w)
        y0, x0 = (h - size) // 2, (w - size) // 2
        cv2.rectangle(display, (x0, y0), (x0 + size, y0 + size), (0, 255, 0), 2)

        status = "RECORDING" if recording else "READY — press SPACE to start"
        color  = (0, 0, 255) if recording else (0, 200, 200)
        cv2.putText(display, f"{gesture.upper()}  {collected}/{target}  {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, "SPACE=start/pause  q=next gesture  ESC=save+quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Inset: 96x96 preview scaled to 192x192
        inset = cv2.resize(crop, (192, 192), interpolation=cv2.INTER_NEAREST)
        inset_bgr = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)
        display[0:192, w-192:w] = inset_bgr

        cv2.imshow("Webcam Collector", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:   # ESC — save and quit everything
            cap.release()
            cv2.destroyAllWindows()
            X, y = save_dataset(dataset_dir, all_images, all_labels)
            counts = {g: int(np.sum(y == label_map[g])) for g in label_map}
            print(f"\nSaved {len(X)} total samples.")
            print(f"Per-class: {counts}")
            exit(0)

        if key == ord('q'):
            break

        # Debounced SPACE toggle — ignores key-repeat
        if key == ord(' '):
            if not space_was_down:
                recording = not recording
                frame_buffer.clear()
                space_was_down = True
        else:
            space_was_down = False

        if not recording:
            continue

        # Quality + diversity checks
        if is_blurry(crop) or is_too_dark(crop):
            rejected += 1
            continue

        if len(frame_buffer) > 0:
            diff = cv2.absdiff(crop, frame_buffer[-1])
            if np.mean(diff) < 5:
                continue  # too similar to last saved frame

        frame_buffer.append(crop.copy())
        all_images.append(crop)
        all_labels.append(label_map[gesture])
        collected += 1

    cv2.destroyAllWindows()
    print(f"  Collected {collected} new samples for '{gesture}' (rejected {rejected})")

# ── Save ───────────────────────────────────────────────────────────────────────

cap.release()
X, y = save_dataset(dataset_dir, all_images, all_labels)
counts = {g: int(np.sum(y == label_map[g])) for g in label_map}

print("\n" + "=" * 70)
print(f"Dataset saved: {len(X)} total samples")
print(f"Per-class    : {counts}")
print(f"Location     : {dataset_dir}")
print(f"\nNext: python train_cnn_model.py  (dataset: {dataset_name})")
print("=" * 70 + "\n")