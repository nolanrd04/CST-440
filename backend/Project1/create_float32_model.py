"""
Create a float32 (non-quantized) TFLite model for testing.
This helps determine if int8 quantization is causing Arduino inference issues.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU

import tensorflow as tf
import numpy as np

print("=" * 60)
print("Creating Float32 TFLite Model for Arduino Testing")
print("=" * 60)

# Generate training data (same as int8 model)
np.random.seed(42)
tf.random.set_seed(42)

x_base = np.linspace(-3.14, 3.14, 5000)

X_samples = []
y_samples = []

for x_val in x_base:
    X_samples.append([x_val, 1, 0])  # sin
    y_samples.append(np.sin(x_val))
    X_samples.append([x_val, 0, 1])  # cos
    y_samples.append(np.cos(x_val))

x = np.array(X_samples)
y = np.array(y_samples)

# Normalize x values to [0, 1]
x_normalized = x.copy()
x_normalized[:, 0] = (x[:, 0] + 3.14) / (2 * 3.14)

# Train/test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=42)

# Same model architecture as int8
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

print(f"Training model ({model.count_params()} parameters)...")
model.fit(x_train, y_train, epochs=100, validation_split=0.2, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])

# Test accuracy
y_pred = model.predict(x_test, verbose=0).flatten()
errors = np.abs(y_test - y_pred)
print(f"Float32 Keras model MAE: {np.mean(errors):.4f}")

# Convert to float32 TFLite (NO quantization)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open('trig_model_float32.tflite', 'wb') as f:
    f.write(tflite_model)
print(f"\nSaved trig_model_float32.tflite ({len(tflite_model)} bytes)")

# Test the TFLite model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"\nInput type: {input_details[0]['dtype']}")
print(f"Output type: {output_details[0]['dtype']}")

# Test sin(-3.14)
test_input = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)  # x_norm=0, is_sin=1, is_cos=0
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])[0][0]
print(f"\nTest sin(-3.14): predicted={output:.4f}, actual={np.sin(-3.14):.4f}")

# Generate C header
header_path = 'trig_inference_arduino/trig_model_float32.h'
with open(header_path, 'w') as f:
    f.write("// Float32 TFLite model for testing (no quantization)\n")
    f.write("#ifndef TRIG_MODEL_FLOAT32_H\n")
    f.write("#define TRIG_MODEL_FLOAT32_H\n\n")
    f.write("alignas(8) const unsigned char trig_model_float32[] = {\n")

    for i in range(0, len(tflite_model), 12):
        row = tflite_model[i:i+12]
        hex_vals = ', '.join(f'0x{b:02x}' for b in row)
        if i + 12 < len(tflite_model):
            f.write(f"  {hex_vals},\n")
        else:
            f.write(f"  {hex_vals}\n")

    f.write("};\n\n")
    f.write(f"const unsigned int trig_model_float32_len = {len(tflite_model)};\n\n")
    f.write("#endif\n")

print(f"Saved {header_path}")
print("\n" + "=" * 60)
print("Now create an Arduino sketch using trig_model_float32.h")
print("If float32 works but int8 doesn't, the issue is int8 handling")
print("=" * 60)