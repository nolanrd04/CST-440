"""
DataPreprocessor.py - Resize, normalize, and augment images for face detection.

Handles:
- Resizing all images to 48x48 grayscale
- Data augmentation on training set (flipping, brightness, noise)
- Normalizing pixel values to [0, 1]
- Saving processed data as .npy files for training
"""

import os
import json
import random
import numpy as np

from DataImporter import import_data, IMG_SIZE

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

LABEL_NAMES = ["non_face", "face"]
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABEL_NAMES)}


def resize_image(image, target_size=IMG_SIZE):
    """Resize an image to target_size x target_size using bilinear interpolation.

    Uses numpy-only interpolation to avoid extra dependencies.
    """
    h, w = image.shape[:2]
    if h == target_size and w == target_size:
        return image

    # Simple bilinear resize using numpy
    row_ratio = h / target_size
    col_ratio = w / target_size

    row_idx = np.floor(np.arange(target_size) * row_ratio).astype(int)
    col_idx = np.floor(np.arange(target_size) * col_ratio).astype(int)

    row_idx = np.clip(row_idx, 0, h - 1)
    col_idx = np.clip(col_idx, 0, w - 1)

    resized = image[np.ix_(row_idx, col_idx)]
    return resized


def augment_flip(image):
    """Randomly flip image horizontally."""
    if random.random() < 0.5:
        return np.fliplr(image)
    return image


def augment_brightness(image, max_delta=0.2):
    """Randomly adjust brightness."""
    delta = random.uniform(-max_delta, max_delta)
    return np.clip(image + delta, 0.0, 1.0)


def augment_noise(image, noise_std=0.02):
    """Add random gaussian noise."""
    noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)


def augment_crop_and_resize(image, min_crop=0.85):
    """Random crop and resize back to original size."""
    h, w = image.shape
    crop_frac = random.uniform(min_crop, 1.0)
    new_h = int(h * crop_frac)
    new_w = int(w * crop_frac)

    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)

    cropped = image[top:top + new_h, left:left + new_w]
    return resize_image(cropped, h)


def preprocess_data():
    """Run the full preprocessing pipeline."""
    print("Importing data...")
    data = import_data()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in ["train", "val", "test"]:
        images = data[split]["images"]
        labels = data[split]["labels"]
        n_samples = len(images)
        print(f"\nProcessing {split} set ({n_samples} samples)...")

        processed = []

        for i in range(n_samples):
            if (i + 1) % 500 == 0 or i == n_samples - 1:
                print(f"  {i + 1}/{n_samples}")

            img = images[i].astype(np.float32)

            # Normalize to [0, 1] if not already
            if img.max() > 1.0:
                img = img / 255.0

            # Resize to target size
            img = resize_image(img, IMG_SIZE)

            # Apply augmentations to training data only
            if split == "train":
                img = augment_flip(img)
                if random.random() < 0.5:
                    img = augment_brightness(img)
                if random.random() < 0.3:
                    img = augment_noise(img)
                if random.random() < 0.3:
                    img = augment_crop_and_resize(img)

            processed.append(img)

        X = np.array(processed, dtype=np.float32)

        # Add channel dimension: (N, 48, 48) -> (N, 48, 48, 1)
        X = X[..., np.newaxis]

        # Compute normalization stats on training set
        if split == "train":
            mean = X.mean()
            std = X.std()
            if std == 0:
                std = 1.0
            np.save(os.path.join(OUTPUT_DIR, "mean.npy"), np.array([mean]))
            np.save(os.path.join(OUTPUT_DIR, "std.npy"), np.array([std]))
        else:
            mean = np.load(os.path.join(OUTPUT_DIR, "mean.npy"))[0]
            std = np.load(os.path.join(OUTPUT_DIR, "std.npy"))[0]

        X = (X - mean) / std

        y = labels.astype(np.int32)

        np.save(os.path.join(OUTPUT_DIR, f"X_{split}.npy"), X)
        np.save(os.path.join(OUTPUT_DIR, f"y_{split}.npy"), y)
        print(f"  Saved X_{split}.npy: {X.shape}, y_{split}.npy: {y.shape}")

    # Save label mapping
    label_map_path = os.path.join(OUTPUT_DIR, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(LABEL_TO_INDEX, f, indent=2)
    print(f"\nLabel mapping saved to {label_map_path}")
    print("Done!")


if __name__ == "__main__":
    preprocess_data()
