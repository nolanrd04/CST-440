import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# NEW ARCHITECTURE: 4 inputs (x + one-hot function selector) -> 1 output
# This allows the model to know which function to compute for each input

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Training model for sin, cos, and tan functions...")
print("=" * 50)

# Generate 5000 points
x_base = np.linspace(-3.14, 3.14, 5000)

# Create training data with one-hot encoding
# Input format: [x_value, is_sin, is_cos, is_tan]
X_samples = []
y_samples = []

# Train all 3 functions
# For sin/cos: use all x values
# For tan: avoid regions near asymptotes (±π/2 ≈ ±1.57) where tan explodes
for x_val in x_base:
    # Always train sin and cos
    X_samples.append([x_val, 1, 0, 0])  # sin
    y_samples.append(np.sin(x_val))
    X_samples.append([x_val, 0, 1, 0])  # cos
    y_samples.append(np.cos(x_val))

    # Only train tan in "safe" regions where |tan(x)| < 3
    # This avoids asymptotes at x ≈ ±1.57, ±4.71
    tan_val = np.tan(x_val)
    if np.abs(tan_val) < 3:
        X_samples.append([x_val, 0, 0, 1])  # tan
        y_samples.append(tan_val)

names = ['sin', 'cos', 'tan']
filename = 'trig_model_all'

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
print(f"Test samples: {len(x_test)}\n")

# Model configuration
hidden_size = 64
epochs = 150

# Model takes 4 inputs [x_normalized, is_sin, is_cos, is_tan] and outputs 1 value
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=(4,)),
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
print("RESULTS")
print("=" * 50)
print(f"Total test samples: {len(x_test)}")
print()

all_accuracies = []

# Map function indices to names
function_map = {1: 'sin', 2: 'cos', 3: 'tan'}

# Evaluate each function separately
for func_idx, name in enumerate(names, start=1):
    # Find samples for this function (check one-hot encoding position)
    # func_idx=1 -> sin (column 1), func_idx=2 -> cos (column 2), func_idx=3 -> tan (column 3)
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

if len(names) > 1:
    print(f"\nOverall: {np.mean(all_accuracies):.1f}% accuracy")
print("=" * 50)

# Save model
model.save(filename, save_format='tf')
print(f"\nModel saved to {filename}")
