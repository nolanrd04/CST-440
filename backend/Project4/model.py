"""
model.py — Project 4: Hand Gesture CNN

Defines build_model(), which returns a compact 2D CNN designed to run
on an Arduino Nano 33 BLE Sense after TFLite int8 conversion.

Size budget (after int8 quantization):
  - Weights : ~50-80 KB  (well under 1 MB Arduino flash)
  - Tensor arena (RAM) : ~50-70 KB  (well under 256 KB Arduino SRAM)

Key choices vs. the naive float32 model:
  - Filters reduced 16→32→64→64  to  8→16→32→32.
    Fewer filters = fewer weight bytes and smaller activation maps.
    5 gesture classes do not need 64-filter capacity.
  - Dense head reduced from 64 to 32 neurons.
  - Int8 quantization (applied in train_cnn_model.py) then shrinks
    every weight and activation by another 4×.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU-only for portability

from tensorflow import keras
from tensorflow.keras import layers


def build_model(img_size: int = 96, num_classes: int = 5) -> keras.Model:
    """
    Builds and returns the gesture-detection CNN (uncompiled).

    Parameters
    ----------
    img_size    : spatial dimension of the square grayscale input (default 96)
    num_classes : number of gesture classes to predict
    """
    inputs = keras.Input(shape=(img_size, img_size, 1), name="input")

    # Block 1 — 96×96 → 48×48
    x = layers.Conv2D(8, 3, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)

    # Block 2 — 48×48 → 24×24
    x = layers.Conv2D(16, 3, padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)

    # Block 3 — 24×24 → 12×12
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D(2, name="pool3")(x)

    # Block 4 — 12×12 → 6×6
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.MaxPooling2D(2, name="pool4")(x)

    # Classifier head  (6×6×32 = 1152 → 32 → num_classes)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(32, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return keras.Model(inputs, outputs, name="gesture_cnn")


if __name__ == "__main__":
    model = build_model()
    model.summary()