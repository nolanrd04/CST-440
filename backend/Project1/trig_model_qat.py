import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
from sklearn.model_selection import train_test_split

# Quantization-Aware Training (QAT) for Trigonometric Model
# This trains the model while simulating int8 quantization, resulting in much better
# accuracy when deployed to microcontrollers compared to post-training quantization.

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Training model with Quantization-Aware Training (QAT)")
print("=" * 50)

# Generate training data (same as original)
x_base = np.linspace(-3.14, 3.14, 5000)

X_samples = []
y_samples = []

for x_val in x_base:
    # Always train sin and cos
    X_samples.append([x_val, 1, 0, 0])  # sin
    y_samples.append(np.sin(x_val))
    X_samples.append([x_val, 0, 1, 0])  # cos
    y_samples.append(np.cos(x_val))

    # Only train tan in "safe" regions where |tan(x)| < 3
    tan_val = np.tan(x_val)
    if np.abs(tan_val) < 3:
        X_samples.append([x_val, 0, 0, 1])  # tan
        y_samples.append(tan_val)

names = ['sin', 'cos', 'tan']
filename = 'trig_model_qat'

# Convert to numpy arrays
x = np.array(X_samples)
y = np.array(y_samples)

# Normalize x values to [0, 1] range
x_normalized = x.copy()
x_normalized[:, 0] = (x[:, 0] + 3.14) / (2 * 3.14)

# Split into train/validation/test (60/20/20)
x_temp, x_test, y_temp, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=42)

print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Test samples: {len(x_test)}\n")

# Model configuration
hidden_size = 64
epochs = 150

# Step 1: Build the model (same architecture as before)
print("Building float32 model...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(hidden_size, activation='relu'),
    tf.keras.layers.Dense(hidden_size // 2, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print(f"Model size: {model.count_params()} parameters")

# Step 2: Apply quantization-aware training
print("\nApplying Quantization-Aware Training...")
quantize_model = tfmot.quantization.keras.quantize_model

# Apply QAT to the entire model
q_aware_model = quantize_model(model)

# Compile the QAT model
q_aware_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print(f"QAT model ready - training with simulated int8 quantization\n")

# Step 3: Train with QAT
print(f"Training for up to {epochs} epochs with early stopping...\n")

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True,
    verbose=1
)

# Train the QAT model
history = q_aware_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    callbacks=[early_stopping],
    verbose=1
)

# Step 4: Evaluate accuracy
print("\n" + "=" * 50)
print("EVALUATING QAT MODEL ACCURACY")
print("=" * 50)

# Test accuracy - separate by function type
y_pred = q_aware_model.predict(x_test, verbose=0).flatten()

tolerance = 0.05

print(f"Total test samples: {len(x_test)}")
print()

all_accuracies = []

# Evaluate each function separately
for func_idx, name in enumerate(names, start=1):
    mask = x_test[:, func_idx] == 1

    if np.sum(mask) > 0:
        func_y_test = y_test[mask]
        func_y_pred = y_pred[mask]

        errors = np.abs(func_y_test - func_y_pred)
        accuracy = np.mean(errors < tolerance) * 100
        mae = np.mean(errors)
        max_error = np.max(errors)

        all_accuracies.append(accuracy)
        print(f"{name}(x): {accuracy:.1f}% accuracy (within {tolerance} absolute error)")
        print(f"  MAE={mae:.6f}, Max Error={max_error:.6f}, Samples={np.sum(mask)}")

if len(names) > 1:
    print(f"\nOverall: {np.mean(all_accuracies):.1f}% accuracy")
print("=" * 50)

# Step 5: Save the QAT model
print(f"\nSaving QAT model to {filename}.keras...")
q_aware_model.save(filename + '.keras', save_format='keras')
print(f"✓ QAT model saved")

# Step 6: Convert to TFLite with int8 quantization
print("\nConverting QAT model to TFLite int8...")

# Create converter from the QAT model
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)

# Enable default optimizations (this will use the QAT information)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

# Save TFLite model
tflite_filename = filename + '.tflite'
with open(tflite_filename, 'wb') as f:
    f.write(tflite_model)

print(f"✓ TFLite model saved: {tflite_filename}")
print(f"  Size: {len(tflite_model)} bytes ({len(tflite_model)/1024:.2f} KB)")

# Step 7: Test TFLite model accuracy
print("\n" + "=" * 50)
print("TESTING TFLite MODEL ACCURACY")
print("=" * 50)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"TFLite Input dtype: {input_details[0]['dtype'].__name__}")
print(f"TFLite Output dtype: {output_details[0]['dtype'].__name__}")

# Test on same test set
tflite_predictions = []
for i in range(len(x_test)):
    input_data = x_test[i:i+1].astype(input_details[0]['dtype'])

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0][0]

    tflite_predictions.append(output)

tflite_predictions = np.array(tflite_predictions)

# Calculate TFLite accuracy
print("\nTFLite Model Accuracy:")
for func_idx, name in enumerate(names, start=1):
    mask = x_test[:, func_idx] == 1

    if np.sum(mask) > 0:
        func_y_test = y_test[mask]
        func_tflite_pred = tflite_predictions[mask]

        errors = np.abs(func_y_test - func_tflite_pred)
        accuracy = np.mean(errors < tolerance) * 100
        mae = np.mean(errors)

        print(f"{name}(x): {accuracy:.1f}% accuracy, MAE={mae:.6f}")

overall_errors = np.abs(y_test - tflite_predictions)
overall_accuracy = np.mean(overall_errors < tolerance) * 100
overall_mae = np.mean(overall_errors)

print(f"\nOverall TFLite Accuracy: {overall_accuracy:.1f}%")
print(f"Overall TFLite MAE: {overall_mae:.6f}")
print("=" * 50)

print("\n✅ Quantization-Aware Training Complete!")
print(f"The TFLite model should now work much better on Arduino!")
print(f"Next step: Run convert_to_tflite.py to generate the C header file.")
