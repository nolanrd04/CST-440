#!/usr/bin/env python3
"""
collect_gesture_data_hagrid.py

Builds a gesture dataset from the local HaGRID sample by:
  - Cropping a square region centered on the hand bbox at CONTEXT_SCALE × the
    bbox size, so the hand fills ~1/CONTEXT_SCALE of the frame with surrounding
    context — similar to holding your hand close to a webcam/Arduino camera
  - Resizing that crop to 96×96 grayscale
  - Applying the same quality checks as collect_gesture_data_improved.py
    (blur rejection, darkness rejection)
  - Saving X.npy / y.npy / label_map.json into data/processed/<dataset_name>/

Usage:
    python collect_gesture_data_hagrid.py
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
HAGRID_ROOT = os.path.join(SCRIPT_DIR, "hagrid-sample-30k-384p")
IMG_ROOT    = os.path.join(HAGRID_ROOT, "hagrid_30k")
ANN_ROOT    = os.path.join(HAGRID_ROOT, "ann_train_val")
DATA_BASE   = os.path.join(SCRIPT_DIR, "data", "processed")

IMG_SIZE       = 96
CONTEXT_SCALE  = 3.0   # crop square = CONTEXT_SCALE × max(bbox_w, bbox_h)
                        # hand fills ~1/CONTEXT_SCALE of the frame (~33%)

VALID_GESTURES = [
    "call", "dislike", "fist", "four", "like", "mute", "ok",
    "one", "palm", "peace", "peace_inverted", "rock", "stop",
    "stop_inverted", "three", "three2", "two_up", "two_up_inverted",
]

# ── UI ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("HAGRID GESTURE DATA COLLECTOR")
print("=" * 70)
print(f"\nHaGRID root : {HAGRID_ROOT}")
print(f"Available gestures: {', '.join(VALID_GESTURES)}\n")

dataset_name = input("Dataset name (e.g., 'hagrid_v1'): ").strip()
if not dataset_name:
    dataset_name = "hagrid_gestures"

gesture_input = input(
    "Gestures to include (comma-separated, or press Enter for all):\n"
    f"  Options: {', '.join(VALID_GESTURES)}\n"
    "  Your choice: "
).strip()

if gesture_input:
    gestures = [g.strip() for g in gesture_input.split(",") if g.strip() in VALID_GESTURES]
else:
    gestures = list(VALID_GESTURES)

if not gestures:
    print("ERROR: No valid gestures specified!")
    exit(1)

samples_str = input("Max samples per gesture (press Enter for all available): ").strip()
max_samples = int(samples_str) if samples_str else None

output_dir = os.path.join(DATA_BASE, dataset_name)
os.makedirs(output_dir, exist_ok=True)

label_map = {g: idx for idx, g in enumerate(gestures)}
with open(os.path.join(output_dir, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)

print(f"\nGestures : {gestures}")
print(f"Max/gesture: {max_samples if max_samples else 'all'}")
print(f"Output   : {output_dir}")
print("=" * 70 + "\n")

# ── Quality helpers (same logic as collect_gesture_data_improved.py) ───────────
def is_blurry(gray, threshold=50):
    if gray is None or gray.size == 0:
        return True
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def is_too_dark(gray, threshold=50):
    if gray is None or gray.size == 0:
        return True
    return np.mean(gray) < threshold

# ── Main loop ──────────────────────────────────────────────────────────────────
all_images = []
all_labels = []

for gesture in gestures:
    ann_path = os.path.join(ANN_ROOT, f"{gesture}.json")
    img_dir  = os.path.join(IMG_ROOT, f"train_val_{gesture}")

    if not os.path.exists(ann_path):
        print(f"  [SKIP] No annotation file for '{gesture}': {ann_path}")
        continue
    if not os.path.isdir(img_dir):
        print(f"  [SKIP] No image directory for '{gesture}': {img_dir}")
        continue

    with open(ann_path) as f:
        annotations = json.load(f)
    image_ids = list(annotations.keys())
    if max_samples:
        rng = np.random.default_rng(42)
        rng.shuffle(image_ids)

    collected = 0
    rejected_blur = 0
    rejected_dark = 0
    rejected_crop = 0

    print(f"Processing '{gesture}' ({len(image_ids)} annotated images) ...")

    for img_id in image_ids:
        if max_samples and collected >= max_samples:
            break

        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue

        ann = annotations[img_id]
        bboxes = ann.get("bboxes", [])
        if not bboxes:
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        h_img, w_img = img_bgr.shape[:2]

        # HaGRID bbox: [x_center, y_center, width, height] normalised to [0,1]
        xc, yc, bw, bh = bboxes[0]
        cx_px = xc * w_img
        cy_px = yc * h_img

        # Square crop at CONTEXT_SCALE × the larger bbox dimension
        half = max(bw * w_img, bh * h_img) * CONTEXT_SCALE / 2
        x1 = int(cx_px - half)
        y1 = int(cy_px - half)
        x2 = int(cx_px + half)
        y2 = int(cy_px + half)

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        if x2 <= x1 or y2 <= y1:
            rejected_crop += 1
            continue

        crop = img_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if is_blurry(gray):
            rejected_blur += 1
            continue
        if is_too_dark(gray):
            rejected_dark += 1
            continue

        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        all_images.append(resized)
        all_labels.append(label_map[gesture])
        collected += 1

        if collected % 100 == 0:
            print(f"  {collected}{(' / ' + str(max_samples)) if max_samples else ''}", end="\r", flush=True)

    print(f"  Collected {collected:>5}  "
          f"(blur={rejected_blur}, dark={rejected_dark}, bad_crop={rejected_crop})")

# ── Save ───────────────────────────────────────────────────────────────────────
X = np.array(all_images, dtype=np.uint8)
y = np.array(all_labels,  dtype=np.int32)

np.save(os.path.join(output_dir, "X.npy"), X)
np.save(os.path.join(output_dir, "y.npy"), y)

print("\n" + "=" * 70)
print("Data collection complete!")
print(f"  Total samples : {len(X)}")
print(f"  Shape         : {X.shape}")
print(f"  Saved to      : {output_dir}")

counts = {g: int(np.sum(y == label_map[g])) for g in gestures}
print(f"  Per-class     : {counts}")
print(f"\nNext: edit train_cnn_model.py → DATASET_NAME = \"{dataset_name}\"")
print("      then: python train_cnn_model.py")
print("=" * 70 + "\n")
