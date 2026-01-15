import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# INT8 QUANTIZED VERSION WITH DERIVED TAN
# Trains only on sin and cos, then computes tan = sin/cos
# This approach leverages high sin/cos accuracy to improve tan predictions

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Training model for sin and cos functions...")
print("tan(x) will be computed as sin(x)/cos(x)")
print("Model will be quantized to INT8 and saved as .tflite")
print("=" * 50)

# Generate 5000 points
x_base = np.linspace(-3.14, 3.14, 5000)

# Create training data with one-hot encoding
# Input format: [x_value, is_sin, is_cos]
X_samples = []
y_samples = []

# Train ONLY sin and cos (tan will be derived)
for x_val in x_base:
    X_samples.append([x_val, 1, 0])  # sin
    y_samples.append(np.sin(x_val))
    X_samples.append([x_val, 0, 1])  # cos
    y_samples.append(np.cos(x_val))

names = ['sin', 'cos', 'tan']  # tan is derived, not trained
filename = 'trig_model_int8.tflite'

# Convert to numpy arrays
x = np.array(X_samples)
y = np.array(y_samples)

# Normalize x values to similar scale as one-hot encoding (0-1 range)
# This helps the network learn more effectively
x_normalized = x.copy()
x_normalized[:, 0] = (x[:, 0] + 3.14) / (2 * 3.14)  # Normalize x to [0, 1]

# Split into train/validation/test (60/20/20)
x_temp, x_test, y_temp, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=42)

print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Test samples: {len(x_test)}")
print(f"Note: Only training on sin and cos (no tan training data)\n")

# Model configuration
hidden_size = 64
epochs = 150

# Model takes 3 inputs [x_normalized, is_sin, is_cos] and outputs 1 value
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(hidden_size, activation='relu'),
    tf.keras.layers.Dense(hidden_size // 2, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

print(f"Model size: {model.count_params()} parameters")
print(f"Training for up to {epochs} epochs with early stopping...\n")

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True,
    verbose=1
)

# Train with validation data and early stopping
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    callbacks=[early_stopping],
    verbose=1
)

# Test accuracy - separate by function type
y_pred = model.predict(x_test, verbose=0).flatten()

tolerance = 0.05

print("\n" + "=" * 50)
print("FLOAT32 MODEL RESULTS (Before Quantization)")
print("=" * 50)
print(f"Total test samples: {len(x_test)}")
print()

all_accuracies = []

# Evaluate sin and cos (trained functions)
for func_idx, name in enumerate(['sin', 'cos'], start=1):
    mask = x_test[:, func_idx] == 1

    if np.sum(mask) > 0:
        func_y_test = y_test[mask]
        func_y_pred = y_pred[mask]

        errors = np.abs(func_y_test - func_y_pred)
        accuracy = np.mean(errors < tolerance) * 100
        mae = np.mean(errors)
        max_error = np.max(errors)

        all_accuracies.append(accuracy)
        print(f"{name}(x): {accuracy:.1f}% accuracy")
        print(f"  MAE={mae:.4f}, Max Error={max_error:.4f}, Samples={np.sum(mask)}")

print(f"\nOverall (sin + cos): {np.mean(all_accuracies):.1f}% accuracy")
print("=" * 50)

# ============================================================================
# EVALUATE TAN USING DERIVED COMPUTATION (sin/cos)
# ============================================================================

print("\n" + "=" * 50)
print("DERIVED TAN EVALUATION (tan = sin/cos)")
print("=" * 50)

# Create test set for tan evaluation
x_tan_test = []
y_tan_test = []

for x_val in x_base:
    tan_val = np.tan(x_val)
    # Only evaluate tan where it's reasonable (avoid asymptotes)
    if np.abs(tan_val) < 3:
        x_tan_test.append(x_val)
        y_tan_test.append(tan_val)

x_tan_test = np.array(x_tan_test)
y_tan_test = np.array(y_tan_test)

# Normalize x values
x_tan_normalized = (x_tan_test + 3.14) / (2 * 3.14)

# Batch predict all sin and cos values at once (much faster than loop)
print("Computing tan predictions from sin/cos...")

# Create batch inputs for all sin predictions
sin_inputs = np.column_stack([x_tan_normalized, np.ones(len(x_tan_normalized)), np.zeros(len(x_tan_normalized))])
sin_preds = model.predict(sin_inputs, verbose=0).flatten()

# Create batch inputs for all cos predictions
cos_inputs = np.column_stack([x_tan_normalized, np.zeros(len(x_tan_normalized)), np.ones(len(x_tan_normalized))])
cos_preds = model.predict(cos_inputs, verbose=0).flatten()

# Compute tan = sin/cos (handle division by zero)
y_tan_pred = np.where(
    np.abs(cos_preds) > 0.01,
    sin_preds / cos_preds,
    np.sign(sin_preds) * 10  # Large value for near-asymptote
)

# Evaluate tan accuracy
errors = np.abs(y_tan_test - y_tan_pred)
accuracy = np.mean(errors < tolerance) * 100
mae = np.mean(errors)
max_error = np.max(errors)

print(f"tan(x): {accuracy:.1f}% accuracy (derived from sin/cos)")
print(f"  MAE={mae:.4f}, Max Error={max_error:.4f}, Samples={len(y_tan_test)}")
print("=" * 50)

# ============================================================================
# INT8 QUANTIZATION AND TFLITE CONVERSION
# ============================================================================

print("\n" + "=" * 50)
print("CONVERTING TO INT8 QUANTIZED TFLITE MODEL")
print("=" * 50)

# Create a representative dataset for quantization calibration
def representative_dataset():
    """Generator function for representative dataset"""
    num_calibration_samples = min(100, len(x_train))
    for i in range(num_calibration_samples):
        yield [x_train[i:i+1].astype(np.float32)]

# Convert to TFLite with INT8 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable INT8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Ensure input and output are also quantized to INT8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print("Quantizing model to INT8...")
tflite_model = converter.convert()

# Save the TFLite model
with open(filename, 'wb') as f:
    f.write(tflite_model)

print(f"✓ INT8 quantized model saved to {filename}")
print(f"  File size: {len(tflite_model) / 1024:.2f} KB")

# ============================================================================
# EVALUATE QUANTIZED MODEL
# ============================================================================

print("\n" + "=" * 50)
print("INT8 QUANTIZED MODEL RESULTS")
print("=" * 50)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=filename)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get quantization parameters
input_scale = input_details[0]['quantization'][0]
input_zero_point = input_details[0]['quantization'][1]
output_scale = output_details[0]['quantization'][0]
output_zero_point = output_details[0]['quantization'][1]

print(f"Input quantization: scale={input_scale:.6f}, zero_point={input_zero_point}")
print(f"Output quantization: scale={output_scale:.6f}, zero_point={output_zero_point}")
print()

# Helper function to run inference with TFLite model
def predict_tflite(x_input):
    """Run inference on quantized TFLite model"""
    input_data = x_input.astype(np.float32)
    input_quantized = (input_data / input_scale + input_zero_point).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], input_quantized)
    interpreter.invoke()

    output_quantized = interpreter.get_tensor(output_details[0]['index'])
    output_dequantized = (output_quantized.astype(np.float32) - output_zero_point) * output_scale

    return output_dequantized[0][0]

# Test sin and cos with quantized model
y_pred_quantized = []
for i in range(len(x_test)):
    input_data = x_test[i:i+1]
    pred = predict_tflite(input_data)
    y_pred_quantized.append(pred)

y_pred_quantized = np.array(y_pred_quantized)

# Evaluate quantized sin and cos
all_accuracies_quantized = []

for func_idx, name in enumerate(['sin', 'cos'], start=1):
    mask = x_test[:, func_idx] == 1

    if np.sum(mask) > 0:
        func_y_test = y_test[mask]
        func_y_pred = y_pred_quantized[mask]

        errors = np.abs(func_y_test - func_y_pred)
        accuracy = np.mean(errors < tolerance) * 100
        mae = np.mean(errors)
        max_error = np.max(errors)

        all_accuracies_quantized.append(accuracy)
        print(f"{name}(x): {accuracy:.1f}% accuracy")
        print(f"  MAE={mae:.4f}, Max Error={max_error:.4f}, Samples={np.sum(mask)}")

print(f"\nOverall (sin + cos): {np.mean(all_accuracies_quantized):.1f}% accuracy")

# ============================================================================
# EVALUATE QUANTIZED TAN (DERIVED)
# ============================================================================

print("\n" + "=" * 50)
print("INT8 QUANTIZED DERIVED TAN RESULTS")
print("=" * 50)

# Compute tan using quantized sin and cos predictions (batched)
print("Computing quantized tan predictions from sin/cos...")

# Batch predict all sin values with quantized model
sin_inputs_q = np.column_stack([x_tan_normalized, np.ones(len(x_tan_normalized)), np.zeros(len(x_tan_normalized))])
sin_preds_q = []
for i in range(len(sin_inputs_q)):
    sin_preds_q.append(predict_tflite(sin_inputs_q[i:i+1]))
sin_preds_q = np.array(sin_preds_q)

# Batch predict all cos values with quantized model
cos_inputs_q = np.column_stack([x_tan_normalized, np.zeros(len(x_tan_normalized)), np.ones(len(x_tan_normalized))])
cos_preds_q = []
for i in range(len(cos_inputs_q)):
    cos_preds_q.append(predict_tflite(cos_inputs_q[i:i+1]))
cos_preds_q = np.array(cos_preds_q)

# Compute tan = sin/cos (handle division by zero)
y_tan_pred_quantized = np.where(
    np.abs(cos_preds_q) > 0.01,
    sin_preds_q / cos_preds_q,
    np.sign(sin_preds_q) * 10
)

# Evaluate quantized tan accuracy
errors = np.abs(y_tan_test - y_tan_pred_quantized)
accuracy = np.mean(errors < tolerance) * 100
mae = np.mean(errors)
max_error = np.max(errors)

print(f"tan(x): {accuracy:.1f}% accuracy (derived from quantized sin/cos)")
print(f"  MAE={mae:.4f}, Max Error={max_error:.4f}, Samples={len(y_tan_test)}")

print("=" * 50)
print("\n✓ Conversion complete! Ready for microcontroller deployment.")
print("✓ To compute tan on microcontroller: predict sin and cos, then divide")