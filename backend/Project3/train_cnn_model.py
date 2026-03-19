"""
train_cnn_model.py - Train a tiny CNN for face detection and export to TFLite.

Handles:
- Loading preprocessed 48x48 grayscale image data
- Building and training a small 2D CNN binary classifier (face vs non-face)
- Evaluating accuracy, precision, recall, F1
- Converting to float32 TFLite model
- Generating a C header for Arduino deployment

Target: >90% accuracy, small enough for Arduino deployment.
"""
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
ARDUINO_DIR = os.path.join(os.path.dirname(__file__), "face_detector_arduino")

NUM_CLASSES = 2
IMG_SIZE = 48


def load_data():
    """Load preprocessed training, validation, and test data."""
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    with open(os.path.join(DATA_DIR, "label_map.json"), "r") as f:
        label_map = json.load(f)

    index_to_label = {v: k for k, v in label_map.items()}

    print(f"Training data:   X={X_train.shape}, y={y_train.shape}")
    print(f"Validation data: X={X_val.shape}, y={y_val.shape}")
    print(f"Test data:       X={X_test.shape}, y={y_test.shape}")
    print(f"  Faces (train):     {np.sum(y_train == 1)}")
    print(f"  Non-faces (train): {np.sum(y_train == 0)}")
    print(f"Label map: {label_map}")

    return X_train, y_train, X_val, y_val, X_test, y_test, label_map, index_to_label


def build_model(input_shape):
    """Build a tiny 2D CNN for face detection (50% smaller model).

    Architecture designed for Arduino deployment:
        Conv2D(8) -> MaxPool -> Conv2D(16) -> MaxPool -> Conv2D(32) -> GAP -> Dense(2)

    This is 50% smaller than the original, reducing model size and memory usage.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        # 48x48x1 -> 24x24x8
        tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # 24x24x8 -> 12x12x16
        tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # 12x12x16 -> 6x6x32
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'),
    ])

    def loss_with_label_smoothing(y_true, y_pred):
        smoothing = 0.1
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), NUM_CLASSES)
        y_true_smooth = y_true_onehot * (1.0 - smoothing) + smoothing / NUM_CLASSES
        return tf.keras.losses.categorical_crossentropy(y_true_smooth, y_pred)

    model.compile(
        optimizer='adam',
        loss=loss_with_label_smoothing,
        metrics=['accuracy'],
    )

    model.summary()
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model with early stopping and learning rate reduction."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    return history


def evaluate_model(model, X_test, y_test, index_to_label):
    """Evaluate the model and print per-class metrics."""
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    if test_acc < 0.90:
        print("WARNING: Accuracy is below 90% target!")
    else:
        print("PASS: Accuracy meets >90% target.")

    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    label_names = [index_to_label[i] for i in range(len(index_to_label))]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=label_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_classes)
    header = "          " + "  ".join(f"{name[:8]:>8}" for name in label_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{label_names[i]:>9} " + "  ".join(f"{val:>8}" for val in row)
        print(row_str)

    return test_acc


def convert_to_tflite(model):
    """Convert Keras model to float32 TFLite format."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_path = os.path.join(os.path.dirname(__file__), "face_model2.0.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"\nSaved TFLite model: {tflite_path} ({len(tflite_model)} bytes)")

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"TFLite input shape:  {input_details[0]['shape']}, dtype: {input_details[0]['dtype']}")
    print(f"TFLite output shape: {output_details[0]['shape']}, dtype: {output_details[0]['dtype']}")
    print("TFLite model verification successful!")

    return tflite_model


def generate_c_header(tflite_model, label_map, index_to_label):
    """Generate C header file for Arduino deployment."""
    os.makedirs(ARDUINO_DIR, exist_ok=True)
    header_path = os.path.join(ARDUINO_DIR, "face_model2.0_data.h")

    mean = np.load(os.path.join(DATA_DIR, "mean.npy")).flatten()
    std = np.load(os.path.join(DATA_DIR, "std.npy")).flatten()

    with open(header_path, 'w') as f:
        f.write("// Face Detection TFLite model and configuration\n")
        f.write("// Auto-generated by train_cnn_model.py - do not edit\n")
        f.write("#ifndef FACE_MODEL_DATA_H\n")
        f.write("#define FACE_MODEL_DATA_H\n\n")

        f.write(f"// Model size: {len(tflite_model)} bytes\n")
        f.write("alignas(8) const unsigned char face_model_tflite[] = {\n")
        for i in range(0, len(tflite_model), 12):
            row = tflite_model[i:i+12]
            hex_vals = ', '.join(f'0x{b:02x}' for b in row)
            if i + 12 < len(tflite_model):
                f.write(f"  {hex_vals},\n")
            else:
                f.write(f"  {hex_vals}\n")
        f.write("};\n\n")
        f.write(f"const unsigned int face_model_tflite_len = {len(tflite_model)};\n\n")

        f.write(f"const int kImageSize = {IMG_SIZE};\n")
        f.write(f"const int kImageChannels = 1;  // grayscale\n\n")

        num_classes = len(label_map)
        f.write(f"const int kNumClasses = {num_classes};\n\n")

        f.write("const char* const kLabelNames[] = {\n")
        for i in range(num_classes):
            name = index_to_label[i]
            comma = "," if i < num_classes - 1 else ""
            f.write(f'  "{name}"{comma}\n')
        f.write("};\n\n")

        f.write("// Pixel normalization: (pixel / 255.0 - mean) / std\n")
        f.write(f"const float kPixelMean = {mean[0]:.6f}f;\n")
        f.write(f"const float kPixelStd = {std[0]:.6f}f;\n\n")

        f.write("#endif\n")

    print(f"Saved C header: {header_path}")


def main():
    print("=" * 60)
    print("Face Detection - CNN Training")
    print("=" * 60)

    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Training on GPU: {gpu_devices[0].name}")
    else:
        print("Training on CPU (no GPU detected)")

    X_train, y_train, X_val, y_val, X_test, y_test, label_map, index_to_label = load_data()

    model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1))

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    train_model(model, X_train, y_train, X_val, y_val)

    test_acc = evaluate_model(model, X_test, y_test, index_to_label)

    print("\n" + "=" * 60)
    print("TFLITE CONVERSION")
    print("=" * 60)
    tflite_model = convert_to_tflite(model)
    generate_c_header(tflite_model, label_map, index_to_label)

    keras_path = os.path.join(os.path.dirname(__file__), "face_model2.0.keras")
    model.save(keras_path)
    print(f"Saved Keras model: {keras_path}")

    print("\n" + "=" * 60)
    print(f"DONE - Test accuracy: {test_acc:.4f}")
    if test_acc >= 0.90:
        print("Target >90% accuracy: ACHIEVED")
    else:
        print("Target >90% accuracy: NOT MET - consider more epochs or data")
    print("=" * 60)


if __name__ == "__main__":
    main()
