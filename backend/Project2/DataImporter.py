"""
DataImporter.py - Load and organize the Speech Commands dataset for wake word recognition.

Handles:
- Loading all WAV file paths grouped by label
- Mapping 5 target words (up, down, off, on, wow) to their labels
- Grouping other word folders into "unknown", subsampled for class balance
- Generating "silence" entries from _background_noise_/ WAVs
- Using validation_list.txt and testing_list.txt for train/val/test splits
"""

import os
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SAMPLE_RATE = 16000
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "archive")

TARGET_WORDS = ["up", "down", "off", "on", "wow"]
BACKGROUND_NOISE_DIR = "_background_noise_"


def load_file_lists():
    """Load the predefined validation and testing file lists."""
    val_path = os.path.join(DATA_DIR, "validation_list.txt")
    test_path = os.path.join(DATA_DIR, "testing_list.txt")

    with open(val_path, "r") as f:
        val_files = set(line.strip() for line in f if line.strip())
    with open(test_path, "r") as f:
        test_files = set(line.strip() for line in f if line.strip())

    return val_files, test_files


def get_word_folders():
    """Get all word category folders (excluding background noise and metadata)."""
    folders = []
    for name in sorted(os.listdir(DATA_DIR)):
        full_path = os.path.join(DATA_DIR, name)
        if os.path.isdir(full_path) and name != BACKGROUND_NOISE_DIR:
            folders.append(name)
    return folders


def generate_silence_paths(num_samples=2400):
    """Generate silence sample entries from background noise files.

    Returns a list of noise file paths that DataPreprocessor will use
    to extract random 1-second segments.
    """
    noise_dir = os.path.join(DATA_DIR, BACKGROUND_NOISE_DIR)
    noise_files = [
        os.path.join(noise_dir, f)
        for f in os.listdir(noise_dir)
        if f.endswith(".wav")
    ]

    silence_entries = []
    for _ in range(num_samples):
        noise_file = random.choice(noise_files)
        silence_entries.append(noise_file)

    return silence_entries


def import_data():
    """Import and organize all data into train/val/test splits.

    Returns:
        dict with keys 'train', 'val', 'test', each containing a list of
        (file_path, label) tuples.
    """
    val_file_set, test_file_set = load_file_lists()
    word_folders = get_word_folders()

    target_files = {"train": [], "val": [], "test": []}
    unknown_files = {"train": [], "val": [], "test": []}

    for folder in word_folders:
        folder_path = os.path.join(DATA_DIR, folder)
        is_target = folder in TARGET_WORDS
        label = folder if is_target else "unknown"

        for wav_file in os.listdir(folder_path):
            if not wav_file.endswith(".wav"):
                continue

            # Use forward slashes to match the format in validation/testing lists
            relative_path = folder + "/" + wav_file
            full_path = os.path.join(DATA_DIR, folder, wav_file)

            if relative_path in val_file_set:
                split = "val"
            elif relative_path in test_file_set:
                split = "test"
            else:
                split = "train"

            entry = (full_path, label)
            if is_target:
                target_files[split].append(entry)
            else:
                unknown_files[split].append(entry)

    # Subsample unknown files to match the average per-class count in each split
    for split in ["train", "val", "test"]:
        avg_target_per_class = len(target_files[split]) // len(TARGET_WORDS)
        random.shuffle(unknown_files[split])
        unknown_files[split] = unknown_files[split][:avg_target_per_class]

    avg_target_per_class = len(target_files["train"]) // len(TARGET_WORDS)

    # Generate silence samples and split 80/10/10
    silence_entries = generate_silence_paths(num_samples=avg_target_per_class)
    random.shuffle(silence_entries)
    n_total = len(silence_entries)
    n_val = n_total // 10
    n_test = n_total // 10
    n_train = n_total - n_val - n_test

    silence_train = [(f, "silence") for f in silence_entries[:n_train]]
    silence_val = [(f, "silence") for f in silence_entries[n_train:n_train + n_val]]
    silence_test = [(f, "silence") for f in silence_entries[n_train + n_val:]]

    # Combine all data
    data = {}
    for split in ["train", "val", "test"]:
        data[split] = target_files[split] + unknown_files[split]

    data["train"] += silence_train
    data["val"] += silence_val
    data["test"] += silence_test

    # Shuffle each split
    for split in data:
        random.shuffle(data[split])

    return data


def print_summary(data):
    """Print a summary of the dataset splits and class distributions."""
    for split in ["train", "val", "test"]:
        entries = data[split]
        label_counts = {}
        for _, label in entries:
            label_counts[label] = label_counts.get(label, 0) + 1

        print(f"\n{split.upper()} set: {len(entries)} samples")
        for label in sorted(label_counts.keys()):
            print(f"  {label:>10s}: {label_counts[label]}")


if __name__ == "__main__":
    data = import_data()
    print_summary(data)
