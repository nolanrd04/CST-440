#!/usr/bin/env python3
"""
test_float32_webcam.py — Test float32 model on webcam

Simple webcam test with the float32 keras model.
Use this FIRST to verify the model works before testing INT8 on Arduino.

Controls:
    'c' : Classify current frame
    'q' : Quit

Usage:
    python test_float32_webcam.py
"""

import os
import json
import cv2
import numpy as np
import tensorflow as tf

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*70)
print("FLOAT32 WEBCAM TEST")
print("="*70)

model_name   = input("Model name: ").strip()
dataset_name = input("Dataset name: ").strip()

MODEL_PATH     = os.path.join(PROJECT_DIR, "models", model_name, f"{model_name}.keras")
LABEL_MAP_PATH = os.path.join(PROJECT_DIR, "data", "processed", dataset_name, "label_map.json")

# Load model
if not os.path.exists(MODEL_PATH):
    print(f"Model not found: {MODEL_PATH}")
    exit(1)

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load label map
if not os.path.exists(LABEL_MAP_PATH):
    print(f"Label map not found: {LABEL_MAP_PATH}")
    exit(1)

with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)

idx_to_name = {v: k for k, v in label_map.items()}
class_names = [idx_to_name[i] for i in range(len(idx_to_name))]

print(f"Classes: {class_names}")
print("\nOpening webcam...")
print("  Controls: 'c' to classify, 'q' to quit\n")

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Could not open webcam!")
    exit(1)

test_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Display frame
    display = frame.copy()
    cv2.putText(display, "Press 'c' to classify, 'q' to quit", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Webcam", display)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('c'):
        # Center-crop the largest square from the frame, then resize to 96×96
        h, w = frame.shape[:2]
        size = min(h, w)
        y0 = (h - size) // 2
        x0 = (w - size) // 2
        square = frame[y0:y0+size, x0:x0+size]
        gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
        resized_norm = resized.astype(np.float32) / 255.0

        # Show the exact pixels the model sees (scaled up so it's visible)
        preview = cv2.resize(resized, (288, 288), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Model input (96x96)", preview)
        
        # Classify
        pred = model.predict(resized_norm[np.newaxis, ..., np.newaxis], verbose=0)[0]
        best_idx = np.argmax(pred)
        best_score = pred[best_idx]
        
        test_count += 1
        print(f"\n{'='*70}")
        print(f"Test #{test_count}")
        print(f"{'='*70}")
        print(f"Scores:")
        for i, score in enumerate(pred):
            print(f"  {class_names[i]:10} : {score:.4f}")
        
        print(f"\n{'→'*35}")
        print(f"PREDICTION: {class_names[best_idx].upper()} ({best_score:.1%})")
        print(f"{'→'*35}\n")
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n✓ Test complete. Ran {test_count} classifications.")
print("="*70 + "\n")
