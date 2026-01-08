import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# change to 2 inputs one output so it can differentiate between sin, cos, tan

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Choose what to learn
print("What function(s) to learn?")
print("1. sin only (easiest)")
print("2. sin and cos (recommended)")
print("3. sin, cos, and tan (hardest)")
choice = input("Choose (1-3, default 2): ") or "2"

# Generate data
x = np.linspace(-3.14, 3.14, 1000).reshape(-1, 1)

if choice == "1":
    y = np.sin(x)
    names = ['sin']
    filename = 'sin_model.keras'
elif choice == "2":
    y = np.column_stack([np.sin(x), np.cos(x)])
    names = ['sin', 'cos']
    filename = 'sin_cos_model.keras'
else:  # choice == "3"
    # For tan, use smaller range to avoid steep gradients
    y = np.column_stack([
        np.sin(x),
        np.cos(x),
        np.clip(np.tan(x), -3, 3)  # Smaller clip range
    ])
    names = ['sin', 'cos', 'tan']
    filename = 'trig_model_all.keras'

# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Build model - size depends on complexity
if choice == "1":
    hidden_size = 32
    epochs = 50
elif choice == "2":
    hidden_size = 48
    epochs = 100
else:
    hidden_size = 64
    epochs = 150

output_size = len(names)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(hidden_size, activation='relu'),
    tf.keras.layers.Dense(output_size)
])

model.compile(optimizer='adam', loss='mse')

print(f"\nModel size: {model.count_params()} parameters")
print(f"Training for {epochs} epochs...\n")

# Train
model.fit(x_train, y_train, epochs=epochs, verbose=1)

# Test accuracy
y_pred = model.predict(x_test, verbose=0)
if len(y_pred.shape) == 1:
    y_pred = y_pred.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

tolerance = 0.05

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)

all_accuracies = []
for i, name in enumerate(names):
    errors = np.abs(y_test[:, i] - y_pred[:, i])
    accuracy = np.mean(errors < tolerance) * 100
    mae = np.mean(errors)
    all_accuracies.append(accuracy)
    print(f"{name}(x): {accuracy:.1f}% accuracy, MAE={mae:.4f}")

if len(names) > 1:
    print(f"\nOverall: {np.mean(all_accuracies):.1f}% accuracy")
print("=" * 50)

# Save model
model.save(filename)
print(f"\nModel saved to {filename}")
