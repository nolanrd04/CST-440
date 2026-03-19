"""
DataImporter.py — Project 4: Gesture Detection

Interactively builds a named dataset from the haGRID-30k sample.
- Prompts for a dataset name (overrides existing if it already exists)
- Prompts for which gestures to include
- Crops each image to its bounding box, pads to square, resizes to 96x96 grayscale
- Saves images and labels as numpy arrays to data/processed/<dataset_name>/

Usage:
    python DataImporter.py
"""

import os
import json
import shutil
import numpy as np
import cv2

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
HAGRID_DIR   = os.path.join(SCRIPT_DIR, "hagrid-sample-30k-384p")
ANN_DIR      = os.path.join(HAGRID_DIR, "ann_train_val")
IMAGE_DIR    = os.path.join(HAGRID_DIR, "hagrid_30k")
OUTPUT_BASE  = os.path.join(SCRIPT_DIR, "data", "processed")

# Inference target size (must match what the Arduino sketch expects)
TARGET_SIZE  = 96
BBOX_PADDING = 0.15   # fractional padding added around each bounding box
MIN_CONF     = 0.80   # drop samples with leading_conf below this
MIN_BOX_PX   = 20     # drop samples whose raw box is smaller than this (too far away)


# ── Helpers ────────────────────────────────────────────────────────────────────

def available_gestures() -> list[str]:
    """Return sorted list of gesture names that have both images and annotations."""
    ann_names   = {f[:-5] for f in os.listdir(ANN_DIR) if f.endswith(".json")}
    image_names = {d.replace("train_val_", "") for d in os.listdir(IMAGE_DIR)
                   if os.path.isdir(os.path.join(IMAGE_DIR, d))}
    return sorted(ann_names & image_names)


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """Letterbox-pad a grayscale or BGR image to a square."""
    h, w = img.shape[:2]
    if h == w:
        return img
    side = max(h, w)
    if img.ndim == 2:
        canvas = np.zeros((side, side), dtype=img.dtype)
    else:
        canvas = np.zeros((side, side, img.shape[2]), dtype=img.dtype)
    top  = (side - h) // 2
    left = (side - w) // 2
    canvas[top:top + h, left:left + w] = img
    return canvas


def crop_hand(image: np.ndarray, bbox_norm: list[float]) -> np.ndarray | None:
    """
    Crop the hand region from image using a normalised COCO bbox [x, y, w, h].
    Returns None if the box is too small to be useful.
    """
    img_h, img_w = image.shape[:2]
    x_n, y_n, w_n, h_n = bbox_norm

    # Convert normalised → pixel coords
    x = int(x_n * img_w)
    y = int(y_n * img_h)
    w = int(w_n * img_w)
    h = int(h_n * img_h)

    if w < MIN_BOX_PX or h < MIN_BOX_PX:
        return None

    # Apply padding
    pad_x = int(w * BBOX_PADDING)
    pad_y = int(h * BBOX_PADDING)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def process_gesture(gesture: str, label_idx: int,
                    images_out: list, labels_out: list) -> int:
    """
    Load all images for a gesture, crop, resize, and append to output lists.
    Returns the number of samples successfully processed.
    """
    ann_path = os.path.join(ANN_DIR, f"{gesture}.json")
    img_folder = os.path.join(IMAGE_DIR, f"train_val_{gesture}")

    with open(ann_path) as f:
        annotations = json.load(f)

    count = 0
    total = len(annotations)

    for i, (uuid, ann) in enumerate(annotations.items()):
        # Progress indicator
        if (i + 1) % 200 == 0 or (i + 1) == total:
            print(f"  [{gesture}] {i + 1}/{total} processed, {count} kept", end="\r")

        # Confidence filter
        if ann.get("leading_conf", 1.0) < MIN_CONF:
            continue

        # Find image file (.jpg or .JPG)
        img_path = os.path.join(img_folder, f"{uuid}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_folder, f"{uuid}.JPG")
        if not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        # Use the first (leading) bounding box
        if not ann.get("bboxes"):
            continue
        crop = crop_hand(image, ann["bboxes"][0])
        if crop is None:
            continue

        # Square-pad → resize → grayscale
        crop = pad_to_square(crop)
        crop = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE),
                          interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        images_out.append(crop)
        labels_out.append(label_idx)
        count += 1

    print()  # newline after \r progress
    return count


# ── Interactive prompts ────────────────────────────────────────────────────────

def prompt_dataset_name() -> str:
    while True:
        name = input("\nEnter a dataset name: ").strip()
        if not name:
            print("  Name cannot be empty.")
            continue
        # Replace spaces with underscores for safe directory names
        name = name.replace(" ", "_")
        out_dir = os.path.join(OUTPUT_BASE, name)
        if os.path.exists(out_dir):
            answer = input(f"  Dataset '{name}' already exists. Override? [y/N]: ").strip().lower()
            if answer == "y":
                shutil.rmtree(out_dir)
                print(f"  Removed existing dataset '{name}'.")
            else:
                continue
        return name


def prompt_gestures(available: list[str]) -> list[str]:
    print("\nAvailable gestures:")
    for i, g in enumerate(available):
        print(f"  [{i + 1:2d}] {g}")

    print("\nEnter gesture numbers separated by commas (e.g. 1,3,5)")
    print("or gesture names separated by commas (e.g. like,fist,palm):")

    while True:
        raw = input("> ").strip()
        if not raw:
            print("  Please select at least one gesture.")
            continue

        parts = [p.strip() for p in raw.split(",")]

        # Detect whether user entered numbers or names
        if all(p.isdigit() for p in parts):
            try:
                chosen = [available[int(p) - 1] for p in parts]
            except IndexError:
                print(f"  Numbers must be between 1 and {len(available)}.")
                continue
        else:
            invalid = [p for p in parts if p not in available]
            if invalid:
                print(f"  Unknown gestures: {', '.join(invalid)}")
                continue
            chosen = parts

        if len(chosen) < 2:
            print("  Select at least 2 gestures.")
            continue

        print(f"\n  Selected: {', '.join(chosen)}")
        confirm = input("  Confirm? [Y/n]: ").strip().lower()
        if confirm in ("", "y"):
            return chosen


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  haGRID Dataset Builder — Project 4")
    print("=" * 55)

    gestures = available_gestures()
    if not gestures:
        print(f"ERROR: No gesture data found in {HAGRID_DIR}")
        return

    dataset_name  = prompt_dataset_name()
    chosen        = prompt_gestures(gestures)

    out_dir = os.path.join(OUTPUT_BASE, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save gesture→label mapping
    label_map = {gesture: idx for idx, gesture in enumerate(chosen)}
    label_map_path = os.path.join(out_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nBuilding dataset '{dataset_name}' with {len(chosen)} classes...")
    print(f"Output → {out_dir}\n")

    all_images: list[np.ndarray] = []
    all_labels: list[int]        = []

    for gesture, idx in label_map.items():
        print(f"Processing: {gesture} (label {idx})")
        n = process_gesture(gesture, idx, all_images, all_labels)
        print(f"  → {n} samples kept\n")

    if not all_images:
        print("ERROR: No samples were processed. Check your dataset path.")
        return

    X = np.array(all_images, dtype=np.uint8)   # shape: (N, 96, 96)
    y = np.array(all_labels, dtype=np.int32)    # shape: (N,)

    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)

    print("=" * 55)
    print(f"  Done! {len(X)} total samples saved.")
    print(f"  X shape : {X.shape}  (uint8, 0-255)")
    print(f"  y shape : {y.shape}")
    print(f"  Classes : {label_map}")
    print(f"  Saved to: {out_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()