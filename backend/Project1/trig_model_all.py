import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate data for all three functions
x = np.linspace(-3.14, 3.14, 1000).reshape(-1, 1)
y = np.column_stack([
    np.sin(x),
    np.cos(x),
    np.clip(np.tan(x), -10, 10)  # Clip tan to avoid infinities
])

# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Build model - outputs all 3 functions
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3)  # 3 outputs: sin, cos, tan
])

model.compile(optimizer='adam', loss='mse')

# Train
print("Training...")
model.fit(x_train, y_train, epochs=100, verbose=1)

# Test accuracy
y_pred = model.predict(x_test, verbose=0)
tolerance = 0.05

print("\n" + "=" * 50)
for i, name in enumerate(['sin', 'cos', 'tan']):
    errors = np.abs(y_test[:, i] - y_pred[:, i])
    accuracy = np.mean(errors < tolerance) * 100
    print(f"{name}(x) - Accuracy: {accuracy:.2f}%, MAE: {np.mean(errors):.4f}")

# Overall
all_errors = np.abs(y_test - y_pred)
overall_acc = np.mean(all_errors < tolerance) * 100
print(f"\nOverall Accuracy: {overall_acc:.2f}%")
print("=" * 50)

# Save model
model.save('trig_model_all.keras')
print("\nModel saved to trig_model_all.keras")