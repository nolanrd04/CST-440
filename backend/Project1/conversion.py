import tensorflow as tf

# Load the .keras model
model = tf.keras.models.load_model('trig_model_all.keras')
print("Model loaded successfully")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
print("Conversion successful")

# Save the TFLite model
with open('trig_model_all.tflite', 'wb') as f:
    f.write(tflite_model)
print("TFLite model saved as trig_model_all.tflite")