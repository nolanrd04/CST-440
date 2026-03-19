from __future__ import annotations

import numpy as np
import tensorflow as tf


def preprocess_frames(frames: np.ndarray, target_hw: tuple[int, int] = (96, 96)) -> np.ndarray:
    """Step 4b scaffold: resize + grayscale for vision gesture input."""
    x = tf.convert_to_tensor(frames)
    x = tf.image.resize(x, target_hw)
    x = tf.image.rgb_to_grayscale(x)
    x = tf.cast(x, tf.float32) / 255.0
    return x.numpy()
