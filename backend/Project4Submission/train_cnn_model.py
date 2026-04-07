"""
train_cnn_model.py — Project 4: Hand Gesture Detection

Loads preprocessed data from data/processed/<DATASET_NAME>/,
trains the CNN defined in model.py, evaluates it, then converts
to a float32 TFLite model and generates a C header for Arduino.

Usage:
    python train_cnn_model.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU-only for portability

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from model import build_model

# ── Config ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_NAME  = input("Dataset name: ").strip()
_model_input  = input("Model name (press Enter for 'gesture_model'): ").strip()
MODEL_NAME    = _model_input if _model_input else "gesture_model"
DATA_DIR      = os.path.join(SCRIPT_DIR, "data", "processed", DATASET_NAME)
MODEL_DIR     = os.path.join(SCRIPT_DIR, "models", MODEL_NAME)
ARDUINO_DIR   = os.path.join(SCRIPT_DIR, "gesture_detector_arduino")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_KERAS   = os.path.join(MODEL_DIR, f"{MODEL_NAME}.keras")
MODEL_TFLITE  = os.path.join(MODEL_DIR, f"{MODEL_NAME}.tflite")
MODEL_HEADER  = os.path.join(MODEL_DIR, "model_data.h")
ARDUINO_HEADER = os.path.join(ARDUINO_DIR, "model_data.h")

SEED          = 42
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.15
EPOCHS        = 30
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
IMG_SIZE      = 96

# ── Helpers ─────────────────────────────────────────────────────────────────────

def load_data():
    X = np.load(os.path.join(DATA_DIR, "X.npy")).astype(np.float32) / 255.0
    y = np.load(os.path.join(DATA_DIR, "y.npy")).astype(np.int32)

    with open(os.path.join(DATA_DIR, "label_map.json")) as f:
        label_map = json.load(f)

    # label_map: {name: idx}  →  we want idx → name for display
    idx_to_name = {v: k for k, v in label_map.items()}
    class_names = [idx_to_name[i] for i in range(len(idx_to_name))]

    # Add channel dim: (N, H, W) → (N, H, W, 1)
    X = X[..., np.newaxis]
    return X, y, class_names


def split_data(X, y):
    """Stratified train / val / test split."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(VAL_SPLIT + TEST_SPLIT), random_state=SEED, stratify=y)

    val_ratio = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1 - val_ratio), random_state=SEED, stratify=y_tmp)

    print(f"  Train : {len(X_train):5d} samples")
    print(f"  Val   : {len(X_val):5d} samples")
    print(f"  Test  : {len(X_test):5d} samples")
    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="train loss")
    axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"],     label="train acc")
    axes[1].plot(history.history["val_accuracy"], label="val acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "training_history.png")
    plt.savefig(path)
    print(f"  Training history plot saved → {path}")


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(path)
    print(f"  Confusion matrix saved → {path}")


def convert_to_tflite(model, X_rep: np.ndarray):
    """
    Convert the Keras model to a full int8 TFLite flatbuffer.

    Int8 quantization reduces both weight storage and tensor arena RAM by ~4×
    compared to float32, which is essential for fitting on the Arduino Nano's
    256 KB SRAM and 1 MB flash.

    A small representative dataset is required so TFLite can compute the
    min/max range of every activation and pick the right int8 scale factors.
    """
    def representative_dataset():
        # Feed ~200 random samples from the training set, one at a time
        indices = np.random.choice(len(X_rep), size=min(200, len(X_rep)), replace=False)
        for i in indices:
            sample = X_rep[i][np.newaxis].astype(np.float32)   # (1, H, W, 1)
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    # Force all ops (including input/output) to int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(MODEL_TFLITE, "wb") as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"  TFLite model saved  → {MODEL_TFLITE}  ({size_kb:.1f} KB)")
    return tflite_model


def generate_c_header(tflite_bytes):
    """Write a C byte-array header suitable for Arduino / TFLite Micro."""
    var_name = "gesture_model_tflite"
    lines = [
        "// Auto-generated — do not edit manually.",
        "// Generated by train_cnn_model.py",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"const unsigned int {var_name}_len = {len(tflite_bytes)};",
        f"alignas(8) const uint8_t {var_name}[] = {{",
    ]

    hex_values = [f"  0x{b:02x}" for b in tflite_bytes]
    for i in range(0, len(hex_values), 12):
        row = hex_values[i:i + 12]
        lines.append(", ".join(row) + ",")

    lines.append("};")

    with open(MODEL_HEADER, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  C header saved      → {MODEL_HEADER}")
    print(f"  To deploy, copy model_data.h to: {ARDUINO_DIR}/")


def verify_tflite(tflite_bytes, X_sample, y_sample):
    """
    Run a quick sanity-check with the TFLite interpreter.

    The int8 model expects int8 input in the range [-128, 127].
    We convert float32 [0, 1] → int8 by scaling to [-128, 127].
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    in_detail  = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    in_idx     = in_detail["index"]
    out_idx    = out_detail["index"]

    # float32 [0,1] → int8 [-128, 127]
    scale, zero_point = in_detail["quantization"]
    if scale == 0:
        scale = 1.0 / 128.0

    preds = []
    for img in X_sample:
        int8_img = (img / scale + zero_point).astype(np.int8)
        interpreter.set_tensor(in_idx, int8_img[np.newaxis])
        interpreter.invoke()
        preds.append(np.argmax(interpreter.get_tensor(out_idx)))

    acc = np.mean(np.array(preds) == y_sample)
    print(f"  TFLite sanity-check accuracy (100 samples): {acc * 100:.1f}%")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("=" * 55)
    print("  Project 4 — Gesture Detection CNN Training")
    print("=" * 55)

    # 1. Load & split data
    print("\n[1/5] Loading data...")
    X, y, class_names = load_data()
    print(f"  Dataset  : {DATASET_NAME}")
    print(f"  Shape    : {X.shape}  dtype={X.dtype}")
    print(f"  Classes  : {class_names}")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 2. Build model
    print("\n[2/5] Building model...")
    num_classes = len(class_names)
    model = build_model(img_size=IMG_SIZE, num_classes=num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # 3. Train
    print("\n[3/5] Training...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    plot_history(history)

    # 4. Evaluate on test set
    print("\n[4/5] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test loss     : {test_loss:.4f}")
    print(f"  Test accuracy : {test_acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\n  Per-class report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    plot_confusion_matrix(y_test, y_pred, class_names)

    # 5. Save & convert
    print("\n[5/5] Saving and converting...")
    model.save(MODEL_KERAS)
    print(f"  Keras model saved   → {MODEL_KERAS}")

    tflite_bytes = convert_to_tflite(model, X_train)
    generate_c_header(tflite_bytes)
    verify_tflite(tflite_bytes, X_test[:100], y_test[:100])

    print("\n" + "=" * 55)
    print("  Training complete.")
    print(f"  Deploy {MODEL_HEADER} to your Arduino sketch.")
    print("=" * 55)


if __name__ == "__main__":
    main()