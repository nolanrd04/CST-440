"""
DataPreprocessor.py - Extract MFCC features from WAV files for wake word recognition.

Handles:
- Reading each WAV file (16kHz, mono)
- Padding or trimming to exactly 16,000 samples (1 second)
- Extracting 13 MFCCs + 13 delta MFCCs (30ms window, 20ms stride) -> shape (49, 26)
- Data augmentation on training audio (time shift, noise injection, SpecAugment)
- Normalizing features (zero mean, unit variance)
- Saving processed data as .npy files for training
"""

import os
import json
import random
import numpy as np
import librosa
import soundfile as sf

from DataImporter import import_data, SAMPLE_RATE

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# MFCC parameters
N_MFCC = 13
N_FEATURES = N_MFCC * 2  # 13 MFCCs + 13 delta MFCCs = 26
HOP_LENGTH = 320    # 20ms stride at 16kHz
N_FFT = 480         # 30ms window at 16kHz
TARGET_LENGTH = SAMPLE_RATE  # 16000 samples = 1 second

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

# Label mapping
LABELS = ["down", "off", "on", "sheila", "silence", "unknown", "up", "wow"]
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}


def load_and_pad_audio(file_path, is_silence=False):
    """Load a WAV file and pad/trim to exactly TARGET_LENGTH samples.

    For silence samples, a random 1-second segment is extracted from the
    longer background noise file.
    """
    audio, sr = sf.read(file_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    if is_silence:
        # Extract a random 1-second segment from the longer noise file
        if len(audio) > TARGET_LENGTH:
            start = random.randint(0, len(audio) - TARGET_LENGTH)
            audio = audio[start:start + TARGET_LENGTH]
        # Scale down silence to realistic background noise level
        audio = audio * 0.3

    # Pad with zeros if too short
    if len(audio) < TARGET_LENGTH:
        padding = TARGET_LENGTH - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')

    # Trim if too long
    if len(audio) > TARGET_LENGTH:
        audio = audio[:TARGET_LENGTH]

    return audio.astype(np.float32)


def extract_mfcc(audio):
    """Extract MFCC + delta MFCC features from audio samples.

    Returns:
        numpy array of shape (49, 26) — 13 MFCCs + 13 delta MFCCs
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    # Compute delta (first derivative) of MFCCs
    delta_mfcc = librosa.feature.delta(mfcc)

    # Concatenate MFCCs and deltas: shape (26, time_frames)
    features = np.concatenate([mfcc, delta_mfcc], axis=0)

    # Transpose to (time_frames, 26)
    features = features.T

    # Ensure consistent shape: pad or trim time axis to 49 frames
    target_frames = 49
    if features.shape[0] < target_frames:
        pad_width = target_frames - features.shape[0]
        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
    elif features.shape[0] > target_frames:
        features = features[:target_frames, :]

    return features.astype(np.float32)


def augment_time_shift(audio, max_shift=1600):
    """Randomly shift audio left/right by up to max_shift samples (~100ms)."""
    shift = random.randint(-max_shift, max_shift)
    return np.roll(audio, shift)


def augment_noise(audio, noise_factor=0.005):
    """Add gaussian noise to audio."""
    noise = np.random.normal(0, noise_factor, audio.shape).astype(np.float32)
    return audio + noise


def augment_spec(features):
    """Apply SpecAugment: randomly zero out time and frequency bands."""
    augmented = features.copy()
    n_time, n_freq = augmented.shape

    # Zero out 1-2 time bands
    for _ in range(random.randint(1, 2)):
        t_width = random.randint(1, 5)
        t_start = random.randint(0, max(0, n_time - t_width))
        augmented[t_start:t_start + t_width, :] = 0

    # Zero out 1-2 frequency bands
    for _ in range(random.randint(1, 2)):
        f_width = random.randint(1, 3)
        f_start = random.randint(0, max(0, n_freq - f_width))
        augmented[:, f_start:f_start + f_width] = 0

    return augmented


def preprocess_data():
    """Run the full preprocessing pipeline."""
    print("Importing data...")
    data = import_data()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split in ["train", "val", "test"]:
        entries = data[split]
        n_samples = len(entries)
        print(f"\nProcessing {split} set ({n_samples} samples)...")

        X = np.zeros((n_samples, 49, N_FEATURES), dtype=np.float32)
        y = np.zeros(n_samples, dtype=np.int32)

        for i, (file_path, label) in enumerate(entries):
            if (i + 1) % 500 == 0 or i == n_samples - 1:
                print(f"  {i + 1}/{n_samples}")

            is_silence = (label == "silence")
            audio = load_and_pad_audio(file_path, is_silence=is_silence)

            # Apply audio augmentations to training data only
            if split == "train":
                if random.random() < 0.5:
                    audio = augment_time_shift(audio)
                if random.random() < 0.5:
                    audio = augment_noise(audio)

            mfcc = extract_mfcc(audio)

            # Apply SpecAugment to training data only
            if split == "train" and random.random() < 0.5:
                mfcc = augment_spec(mfcc)

            X[i] = mfcc
            y[i] = LABEL_TO_INDEX[label]

        # Normalize: zero mean, unit variance per feature (computed on train set)
        if split == "train":
            mean = X.mean(axis=(0, 1), keepdims=True)
            std = X.std(axis=(0, 1), keepdims=True)
            std[std == 0] = 1.0  # avoid division by zero
            # Save normalization stats for inference
            np.save(os.path.join(OUTPUT_DIR, "mean.npy"), mean)
            np.save(os.path.join(OUTPUT_DIR, "std.npy"), std)
        else:
            # Use train stats for val/test normalization
            mean = np.load(os.path.join(OUTPUT_DIR, "mean.npy"))
            std = np.load(os.path.join(OUTPUT_DIR, "std.npy"))

        X = (X - mean) / std

        # Save
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
