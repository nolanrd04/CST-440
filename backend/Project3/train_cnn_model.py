"""
train_cnn_model.py - Train a face detector using a GAN, then export to TFLite.

The discriminator IS the face detection CNN that gets deployed to Arduino.

Phase 1 - GAN Training:
    Train a DCGAN on face-only images. The generator learns to produce faces,
    while the discriminator (our tiny CNN) learns deep features for recognizing
    what a face looks like.

Phase 2 - Fine-tuning:
    Swap the discriminator's final sigmoid(1) head for a softmax(2) head
    (face vs non-face), freeze early conv layers, and fine-tune on the full
    labeled dataset. This transfers the GAN-learned features into a classifier.

Phase 3 - Export:
    Evaluate on test set, convert to TFLite, generate C header for Arduino.

Target: >90% accuracy, small enough for Arduino deployment.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU (RTX 5070 Ti not supported yet)

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
ARDUINO_DIR = os.path.join(os.path.dirname(__file__), "face_detector_arduino")

NUM_CLASSES = 2
IMG_SIZE = 48
LATENT_DIM = 64

# GAN training config
GAN_EPOCHS = 150
GAN_BATCH_SIZE = 64

# Fine-tuning config
FT_EPOCHS = 100
FT_BATCH_SIZE = 64


# ============================================================
# Data loading
# ============================================================

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


def scale_to_tanh(images):
    """Scale images to [-1, 1] for GAN's tanh output."""
    img_min = images.min()
    img_max = images.max()
    if img_max - img_min == 0:
        return np.zeros_like(images), img_min, img_max
    return 2.0 * (images - img_min) / (img_max - img_min) - 1.0, img_min, img_max


# ============================================================
# Phase 1: GAN architecture & training
# ============================================================

def build_generator(latent_dim):
    """Build the DCGAN generator.

    Maps latent vector (64,) -> (48, 48, 1) grayscale image.
    Only used during training — NOT deployed to Arduino.
    """
    model = tf.keras.Sequential([
        layers.Input(shape=(latent_dim,)),

        # Project and reshape: 64 -> 6*6*64
        layers.Dense(6 * 6 * 64, use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Reshape((6, 6, 64)),

        # 6x6x64 -> 12x12x32
        layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        # 12x12x32 -> 24x24x16
        layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        # 24x24x16 -> 48x48x1
        layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        layers.Activation('tanh'),
    ])

    return model


def build_discriminator(img_shape):
    """Build the DCGAN discriminator — this IS the face detector CNN.

    Maps (48, 48, 1) -> real/fake probability.
    After GAN training, the conv layers are reused for classification.

    Architecture (Arduino-sized):
        Conv2D(16) -> Conv2D(32) -> Conv2D(64) -> Flatten -> Dense(1)
    """
    model = tf.keras.Sequential([
        layers.Input(shape=img_shape),

        # 48x48x1 -> 24x24x16
        layers.Conv2D(16, (4, 4), strides=(2, 2), padding='same', name='conv1'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        # 24x24x16 -> 12x12x32
        layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', name='conv2'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        # 12x12x32 -> 6x6x64
        layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', name='conv3'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        # GAN head (replaced after training)
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid', name='gan_head'),
    ])

    return model


@tf.function
def gan_train_step(generator, discriminator, g_optimizer, d_optimizer,
                   loss_fn, real_images, batch_size, latent_dim):
    """Single GAN training step: update discriminator then generator."""
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)

        # Label smoothing for training stability
        real_labels = tf.ones_like(real_output) * 0.9
        fake_labels = tf.zeros_like(fake_output) + 0.1

        d_loss_real = loss_fn(real_labels, real_output)
        d_loss_fake = loss_fn(fake_labels, fake_output)
        d_loss = d_loss_real + d_loss_fake

        g_loss = loss_fn(tf.ones_like(fake_output), fake_output)

    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)

    d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

    return d_loss, g_loss


def train_gan(face_images):
    """Phase 1: Train the GAN on face images only.

    The discriminator learns what real faces look like by trying to
    distinguish them from generator fakes. These learned conv features
    transfer directly to the classification task.

    Returns:
        Trained discriminator (our CNN), generator (for sample visualization)
    """
    print("Scaling face images to [-1, 1] for GAN training...")
    scaled, _, _ = scale_to_tanh(face_images)

    dataset = tf.data.Dataset.from_tensor_slices(scaled)
    dataset = dataset.shuffle(len(scaled)).batch(GAN_BATCH_SIZE, drop_remainder=True)

    img_shape = (IMG_SIZE, IMG_SIZE, 1)
    generator = build_generator(LATENT_DIM)
    discriminator = build_discriminator(img_shape)

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    print(f"\nGenerator params:     {generator.count_params()}")
    print(f"Discriminator params: {discriminator.count_params()}")

    for epoch in range(GAN_EPOCHS):
        d_losses, g_losses = [], []

        for batch in dataset:
            d_loss, g_loss = gan_train_step(
                generator, discriminator, g_optimizer, d_optimizer,
                loss_fn, batch, batch.shape[0], LATENT_DIM,
            )
            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))

        if (epoch + 1) % 25 == 0 or epoch == 0:
            avg_d = np.mean(d_losses)
            avg_g = np.mean(g_losses)
            print(f"  Epoch {epoch + 1:3d}/{GAN_EPOCHS} - D loss: {avg_d:.4f}, G loss: {avg_g:.4f}")

    # Save a grid of generated faces for inspection
    noise = tf.random.normal([16, LATENT_DIM])
    fake = generator(noise, training=False).numpy()
    save_sample_grid(fake, os.path.join(os.path.dirname(__file__), "gan_generated_faces.png"),
                     "GAN-Generated Faces (Phase 1)")

    return discriminator, generator


def save_sample_grid(images, filename, title, n=8):
    """Save a grid of sample images for visual inspection."""
    rows = min(2, (len(images) + n - 1) // n)
    fig, axes = plt.subplots(rows, n, figsize=(n * 1.5, rows * 1.5))
    fig.suptitle(title, fontsize=12)
    if rows == 1:
        axes = [axes]
    for i in range(min(rows * n, len(images))):
        ax = axes[i // n][i % n]
        ax.imshow(images[i, :, :, 0], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"  Saved: {filename}")


# ============================================================
# Phase 2: Convert discriminator to classifier & fine-tune
# ============================================================

def discriminator_to_classifier(discriminator):
    """Replace the discriminator's GAN head with a classification head.

    Keeps all conv layers (with GAN-learned weights), replaces:
        Flatten -> Dense(1, sigmoid)       [GAN head]
    with:
        GlobalAveragePooling2D -> Dropout -> Dense(2, softmax)  [classifier]

    Freezes early conv layers so GAN features are preserved,
    only fine-tunes the last conv layer + new classification head.
    """
    # Extract conv layers (everything before Flatten)
    # The discriminator structure is: conv1, leaky, drop, conv2, leaky, drop, conv3, leaky, drop, flatten, dense
    conv_layers = []
    for layer in discriminator.layers:
        if isinstance(layer, layers.Flatten):
            break
        conv_layers.append(layer)

    # Build the classifier using the discriminator's conv backbone
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = inp
    for layer in conv_layers:
        x = layer(x)

    # New classification head (replaces Flatten + Dense(1))
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(NUM_CLASSES, activation='softmax', name='classifier_head')(x)

    classifier = tf.keras.Model(inputs=inp, outputs=x)

    # Freeze early conv layers (conv1, conv2) — only train conv3 + head
    for layer in classifier.layers:
        if hasattr(layer, 'name') and layer.name in ('conv1', 'conv2'):
            layer.trainable = False

    trainable = sum(p.numpy().size for p in classifier.trainable_weights)
    frozen = sum(p.numpy().size for p in classifier.non_trainable_weights)
    print(f"\nClassifier: {trainable} trainable params, {frozen} frozen params")

    def sparse_crossentropy_with_label_smoothing(y_true, y_pred):
        smoothing = 0.1
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), NUM_CLASSES)
        y_true_smooth = y_true_onehot * (1.0 - smoothing) + smoothing / NUM_CLASSES
        return tf.keras.losses.categorical_crossentropy(y_true_smooth, y_pred)

    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=sparse_crossentropy_with_label_smoothing,
        metrics=['accuracy'],
    )

    classifier.summary()
    return classifier


def fine_tune(classifier, X_train, y_train, X_val, y_val):
    """Fine-tune the classifier on labeled face/non-face data."""
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

    history = classifier.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=FT_EPOCHS,
        batch_size=FT_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    return history


# ============================================================
# Phase 3: Evaluate & export
# ============================================================

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

    tflite_path = os.path.join(os.path.dirname(__file__), "face_model.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"\nSaved TFLite model: {tflite_path} ({len(tflite_model)} bytes)")

    # Verify
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
    header_path = os.path.join(ARDUINO_DIR, "face_model_data.h")

    mean = np.load(os.path.join(DATA_DIR, "mean.npy")).flatten()
    std = np.load(os.path.join(DATA_DIR, "std.npy")).flatten()

    with open(header_path, 'w') as f:
        f.write("// Face Detection TFLite model and configuration\n")
        f.write("// Discriminator from GAN, fine-tuned for face/non-face classification\n")
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


# ============================================================
# Main pipeline
# ============================================================

def main():
    print("=" * 60)
    print("Face Detection - GAN-trained CNN")
    print("=" * 60)
    print("The discriminator IS the deployed model.")
    print("Phase 1: GAN training (learn face features)")
    print("Phase 2: Fine-tune discriminator (face vs non-face)")
    print("Phase 3: Export to TFLite for Arduino")
    print("=" * 60)

    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"Training on GPU: {len(gpu_devices)} GPU(s) detected")
    else:
        print("Training on CPU (GPU not available or disabled)")

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, label_map, index_to_label = load_data()

    # --- Phase 1: GAN training on face images only ---
    print("\n" + "=" * 60)
    print("PHASE 1: GAN TRAINING (face images only)")
    print("=" * 60)

    face_images = X_train[y_train == 1]
    print(f"Training GAN on {len(face_images)} face images...")

    discriminator, generator = train_gan(face_images)

    # --- Phase 2: Convert discriminator to classifier & fine-tune ---
    print("\n" + "=" * 60)
    print("PHASE 2: FINE-TUNING DISCRIMINATOR AS CLASSIFIER")
    print("=" * 60)
    print("Replacing GAN head (sigmoid) with classifier head (softmax)...")
    print("Freezing conv1, conv2 — fine-tuning conv3 + classifier head...")

    classifier = discriminator_to_classifier(discriminator)
    fine_tune(classifier, X_train, y_train, X_val, y_val)

    # --- Phase 3: Evaluate & export ---
    test_acc = evaluate_model(classifier, X_test, y_test, index_to_label)

    print("\n" + "=" * 60)
    print("PHASE 3: TFLITE CONVERSION & EXPORT")
    print("=" * 60)
    tflite_model = convert_to_tflite(classifier)
    generate_c_header(tflite_model, label_map, index_to_label)

    keras_path = os.path.join(os.path.dirname(__file__), "face_model.keras")
    classifier.save(keras_path)
    print(f"Saved Keras model: {keras_path}")

    print("\n" + "=" * 60)
    print(f"DONE - Test accuracy: {test_acc:.4f}")
    if test_acc >= 0.90:
        print("Target >90% accuracy: ACHIEVED")
    else:
        print("Target >90% accuracy: NOT MET - consider more GAN epochs or data")
    print("=" * 60)


if __name__ == "__main__":
    main()
