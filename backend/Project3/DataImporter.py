"""
DataImporter.py - Download and organize face/non-face image data.

Handles:
- Downloading LFW (Labeled Faces in the Wild) dataset for face samples
- Downloading CIFAR-10 for non-face samples (animals, vehicles, objects)
- Organizing into train/val/test splits with class balance
"""

import os
import random
import numpy as np
from sklearn.datasets import fetch_lfw_people

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

# Image size for the model input (grayscale)
IMG_SIZE = 48


def download_face_data(min_faces_per_person=10):
    """Download LFW face dataset via sklearn.

    Returns:
        numpy array of face images, shape (N, 62, 47) uint8-range floats
    """
    print("Downloading LFW face dataset...")
    lfw = fetch_lfw_people(
        min_faces_per_person=min_faces_per_person,
        resize=0.5,
        color=False,
    )
    images = lfw.images  # shape (N, 62, 47), float64 in [0, 255]
    print(f"  Loaded {len(images)} face images, shape {images[0].shape}")
    return images


def download_nonface_data(num_samples=3000):
    """Download CIFAR-10 and extract non-person images.

    Uses classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    All CIFAR-10 classes are non-face, so we use all of them.

    Returns:
        numpy array of non-face images, shape (N, 32, 32) grayscale
    """
    import tensorflow as tf

    print("Downloading CIFAR-10 for non-face samples...")
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

    # Combine train and test
    all_images = np.concatenate([x_train, x_test], axis=0)

    # Convert to grayscale: weighted sum of RGB channels
    gray = (
        0.2989 * all_images[:, :, :, 0]
        + 0.5870 * all_images[:, :, :, 1]
        + 0.1140 * all_images[:, :, :, 2]
    )

    # Subsample to desired count
    indices = np.random.permutation(len(gray))[:num_samples]
    gray = gray[indices]

    print(f"  Loaded {len(gray)} non-face images, shape {gray[0].shape}")
    return gray


def import_data():
    """Import face and non-face data, split into train/val/test.

    Returns:
        dict with keys 'train', 'val', 'test', each containing:
            'images': numpy array of images
            'labels': numpy array of labels (1=face, 0=non-face)
    """
    face_images = download_face_data()
    num_faces = len(face_images)

    # Get roughly equal number of non-face samples
    nonface_images = download_nonface_data(num_samples=num_faces)
    num_nonfaces = len(nonface_images)

    print(f"\nDataset: {num_faces} faces, {num_nonfaces} non-faces")

    # Create labels
    face_labels = np.ones(num_faces, dtype=np.int32)
    nonface_labels = np.zeros(num_nonfaces, dtype=np.int32)

    # Shuffle each class independently before splitting
    face_perm = np.random.permutation(num_faces)
    face_images = face_images[face_perm]

    nonface_perm = np.random.permutation(num_nonfaces)
    nonface_images = nonface_images[nonface_perm]

    # Split each class: 80% train, 10% val, 10% test
    def split_data(images, labels):
        n = len(images)
        n_val = n // 10
        n_test = n // 10
        n_train = n - n_val - n_test

        splits = {
            "train": (images[:n_train], labels[:n_train]),
            "val": (images[n_train:n_train + n_val], labels[n_train:n_train + n_val]),
            "test": (images[n_train + n_val:], labels[n_train + n_val:]),
        }
        return splits

    face_splits = split_data(face_images, face_labels)
    nonface_splits = split_data(nonface_images, nonface_labels)

    data = {}
    for split in ["train", "val", "test"]:
        images = np.concatenate([face_splits[split][0], nonface_splits[split][0]])
        labels = np.concatenate([face_splits[split][1], nonface_splits[split][1]])

        # Shuffle combined data
        perm = np.random.permutation(len(images))
        data[split] = {
            "images": images[perm],
            "labels": labels[perm],
        }

    return data


def print_summary(data):
    """Print a summary of the dataset splits and class distributions."""
    label_names = {0: "non-face", 1: "face"}
    for split in ["train", "val", "test"]:
        images = data[split]["images"]
        labels = data[split]["labels"]
        print(f"\n{split.upper()} set: {len(images)} samples")
        for cls in [0, 1]:
            count = np.sum(labels == cls)
            print(f"  {label_names[cls]:>10s}: {count}")


if __name__ == "__main__":
    data = import_data()
    print_summary(data)
