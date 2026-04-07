# Project 4: Hand Gesture Detection (ArduCAM + TensorFlow Lite)

Real-time hand gesture recognition on Arduino Nano 33 BLE Sense with ArduCAM OV2640 camera.

**Status:** Model training issue identified, root cause diagnosed, solutions provided.

---

## Development History & Evolution

### Initial Approach (v1): Five Similar Gestures
**Problem:** Started with five gestures: **call, like, dislike, ok, mute**
- These gestures were visually too similar to distinguish reliably
- Model achieved only ~20% confidence on Arduino, essentially guessing uniformly across 5 classes (20% per gesture)
- Even on webcam testing, accuracy was poor due to gesture ambiguity

**Resolution:** Completely redesigned dataset with more distinct gestures

### Dataset Remake (v2): New Gestures with Improved Cropping
**Changes Made:**
1. **New gesture set:** Switched to more distinctive gestures
   - **Palm** - open hand, fingers spread (high contrast)
   - **Fist** - closed hand (clear shape)
   - **Dislike** - thumbs down (unique silhouette)
   - **Peace** - two fingers up (distinctive)
   - **Call** - hand to ear (specific pose)

2. **Improved hand cropping & padding:**
   - Recollected dataset with consistent hand positioning
   - Applied better image cropping strategies to isolate hands from background
   - Added consistent padding to standardize input size
   - Resulted in cleaner, more uniform training data

3. **Custom data augmentation:**
   - Supplemented dataset with manually captured hand positions
   - Added consistent hand positioning examples to arrays
   - Ensured variety in distance, angle, and lighting during collection
   - Targeted ~200-500 samples per gesture for better generalization

### Results After Retraining

#### Arduino Testing (INT8 Quantized Model)
- Initial Arduino results improved from uniform 20% confidence but still showed limitations
- Realized quantization was dropping model accuracy significantly
- INT8 quantization was destroying model precision needed for gesture distinction

#### Webcam Testing (Float32 Model)  
**Strong performers:**
- ✅ **Palm** - Excellent recognition, robust across variations
- ✅ **Fist** - Excellent recognition, very distinctive shape
- ✅ **Dislike** - Excellent recognition, unique pose

**Moderate performers:**
- ⚠️ **Peace** - Good recognition, but **only works with specific framing** (requires fingers to be clearly separated and visible)

**Weak performer:**
- ❌ **Call** - Poor recognition, struggles with consistency and hand pose variations

---

## Quick Start

### 1. Setup
```bash
cd Project4
python3.11 -m venv .venv #more compatible with tensorflow
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Collect Training Data
```bash
python collect_training_data.py
# OR (better quality w/ filtering) 
python collect_gesture_data_improved.py
```

### 3. Train Model
```bash
python train_cnn_model.py
# OR (recommended: quantization-aware training)
python train_cnn_model_qat.py
```

### 4. Deploy to Arduino
- Copy `gesture_model_data.h` (or `gesture_model_data_qat.h`) to `gesture_detector_arduino/`
- Open `gesture_detector_arduino/gesture_detector_arduino.ino` in Arduino IDE
- Upload to board

### 5. Test
- Open Serial Monitor (115200 baud)
- Send character 'c' to capture & classify

---

## File Structure

```
Project4/
├── gesture_detector_arduino/
│   ├── gesture_detector_arduino.ino    # Main sketch
│   ├── model_data.h                    # TFLite model (auto-generated)
│   └── platformio.ini
├── train_cnn_model.py                  # Standard training
├── train_cnn_model_qat.py             # Quantization-aware training (better)
├── collect_training_data.py            # Data collection
├── collect_gesture_data_improved.py   # Better data collection w/ quality checks
├── model.py                            # CNN architecture
├── analyze_model_issue.py              # Diagnostic tool (run to analyze model)
├── MODEL_ISSUE_ANALYSIS.md            # Detailed solutions guide
└── data/processed/my_gestures/        # Dataset
```

---

## Troubleshooting: Model Predicts "No Gesture" or Always "Call"

### Root Cause Analysis

**April 3, 2026: Systematic Diagnosis**

Ran diagnostic tool (`analyze_model_issue.py`) to investigate poor gesture detection performance.

#### Finding 1: Class Balance ✅ OK
```
Dataset: 250 total samples (50 per gesture)
Classes: call (50), dislike (50), like (50), mute (50), ok (50)
Balance ratio: 1.00:1 (perfect)
→ NOT a class imbalance issue
```

#### Finding 2: Float32 Performance ✅ EXCELLENT
```
Model accuracy before quantization: 90.4% mean confidence ✅
Min: 0.47  |  Max: 1.00  |  Std: 0.17
→ Model is actually very good in float32 format
```

#### Finding 3: INT8 Performance ❌ CATASTROPHIC
```
Model accuracy after int8 quantization: 36.5% mean confidence ❌❌❌
Min: 0.25  |  Max: 0.40  |  Std: 0.05
Confidence loss: 0.54 (59.7% degradation)
→ INT8 CONVERSION DESTROYED THE MODEL ACCURACY
```

**Key Observation:** Per-class confidence (INT8):
- **dislike:** 0.40 (best performer) ← Model does well here
- **call:** 0.33
- **like:** 0.29
- **mute:** 0.31
- **ok:** 0.30

Asymmetric performance suggests training data quality varies by gesture.

#### Finding 4: Threshold Not The Issue ❌
User test: Lowered threshold from 0.60 → 0.30
- **Result:** NO CHANGE in behavior (still mostly "call" or "no gesture")
- **Conclusion:** Problem is model confidence, not threshold
- Even at 0.30, most predictions below threshold, causing repeated "no gesture"

#### Finding 5: Dataset Size ⚠️
```
Current: 250 samples (50/gesture)
Industry standard: 300-500 per gesture minimum
Issue: Small dataset + quantization = poor generalization
```

### Root Cause: Post-Training Quantization

The model is trained as float32, then quantized to int8 **after** training. This is suboptimal because:
- Model never learned to work with int8 constraints during training
- Quantization is lossy — 59.7% accuracy loss is severe but fixable

### Solutions

#### Option 1: Retrain with Quantization-Aware Training (QAT) [RECOMMENDED]
**Time:** 1-2 hours | **Expected INT8 confidence:** 0.50-0.55 ✅

```bash
pip install tensorflow-model-optimization
python train_cnn_model_qat.py
```

**Why it works:** Model learns with int8 quantization in mind from the start, recovering 20-30% of lost confidence.

#### Option 2: Collect Better Data + QAT [BEST]
**Time:** 2-4 hours | **Expected INT8 confidence:** 0.70-0.80 ✅✅

```bash
python collect_gesture_data_improved.py  # Collects 300-500 samples per gesture
# Follow prompts, ensure varied backgrounds/lighting/distances
python train_cnn_model_qat.py
```

**Impact:** More data + better training approach = significantly better model.

#### Option 3: Test Float32 First (Diagnostic)
Verify the issue is quantization-specific:

```bash
python test_float32_webcam.py  # Simple script (recommended)
```

Or use this inline test:
```bash
python << 'EOF'
import cv2, json, numpy as np, tensorflow as tf
model = tf.keras.models.load_model("gesture_model.keras")
with open("data/processed/my_gestures/label_map.json") as f:
    label_map = json.load(f)
idx_to_name = {v: k for k, v in label_map.items()}
class_names = [idx_to_name[i] for i in range(len(idx_to_name))]

cap = cv2.VideoCapture(0)
print("Float32 Webcam Test (press 'c' to classify, 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_norm = cv2.resize(gray, (96, 96)).astype(np.float32) / 255.0
    cv2.imshow("Webcam Test", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('c'):
        pred = model.predict(resized_norm[np.newaxis, ..., np.newaxis], verbose=0)[0]
        for i, score in enumerate(pred):
            print(f"  {class_names[i]:10}: {score:.4f}")
        print(f"  → BEST: {class_names[np.argmax(pred)]} ({pred.max():.1%})\n")
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
EOF
```

**Test Results (159 runs):**
```
✓ call → 100% accuracy (any hand position)
✓ dislike → 70% accuracy (when covering face)
✓ ok → 75-99% accuracy (left hand positions)
✗ like → 0% accuracy (always predicted as call/ok)
✗ mute → 0% accuracy (always predicted as call)
```

**Finding:** Float32 model also fails on like/mute → **NOT a quantization issue**, it's **training data distinctiveness**. Like and mute gestures are not distinctive enough in the training set (only 50 samples each with limited variation).

**Action:** Recollect like & mute with 300-400 diverse samples each, then retrain. See [POTENTIAL_ISSUE.md](POTENTIAL_ISSUE.md).

---

## Camera & Preprocessing Details

### Cropping Strategy (Important!)
**Current implementation:** ✅ CORRECT
```cpp
// gesture_detector_arduino.ino, lines 100-102
Crop 192×192 centered region from 320×240 image
    ↓
Downsample to 96×96 by sampling every 2nd pixel
```

**Why this approach:**
- ✅ Removes edge noise/black borders from camera
- ✅ Preserves detail (192→96 = 2× downsampling is gentle)
- ❌ Don't just downsample 320×240→96×96 (loses too much resolution)

**Implementation details:**
```cpp
const int CROP_ROW_OFFSET = (240 - 192) / 2 = 24   // Skip rows 0-23
const int CROP_COL_OFFSET = (320 - 192) / 2 = 64   // Skip cols 0-63
```

### Camera Exposure Control (OV2640 Registers)
```cpp
// gesture_detector_arduino.ino, lines 161-163
cam.wrSensorReg8_8(0xFF, 0x01);
cam.wrSensorReg8_8(0x13, 0x00);  // COM8: Disable auto-exposure (manual)
cam.wrSensorReg8_8(0x14, 0x48);  // COM9: AGC gain = 32x
```

**Tuning:**
- Lower `0x14` value → darker image (less visible)
- Higher `0x14` value → brighter image (more washed out)
- Current `0x48` = good baseline, adjust if needed

### Debug: View Raw Preprocessed Images

The Arduino sketch outputs the grayscale image as hex after each inference:

```
Scores:
  call: 0.33
  ...
DEBUG_IMAGE_START
2f 3a 4b 5c 6d ... (hex bytes representing 96×96 image)
DEBUG_IMAGE_END
```

**Parse with Python to visualize what the model sees:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Copy hex output from serial monitor
hex_data = "2f 3a 4b 5c 6d ..."  
image = np.frombuffer(bytes.fromhex(hex_data.replace(" ", "")), dtype=np.uint8)
image = image.reshape(96, 96)

plt.imshow(image, cmap='gray')
plt.title("Model Input (96×96 Grayscale)")
plt.savefig('debug_frame.png')
plt.show()
```

Use this to verify camera framing and preprocessing are working correctly.

---

## Dataset Recommendations

**Current state:** 250 samples (50/gesture)  
**Minimum viable:** 250 samples  
**Recommended:** 1000+ samples (200-500 per gesture)

**Collection tips for better data:**
- **Backgrounds:** White wall (training habitat), desk, outdoors, various
- **Lighting:** Bright office, dim room, natural sunlight, artificial lamps
- **Distance:** 20cm (close), 30cm (normal), 50cm (far)
- **Variations:** Different hand sizes, skin tones, ages if possible

**Why this matters:**  
Current dataset likely biased toward specific collection conditions. Quantization is unforgiving with narrow training distribution. Broader data = better generalization.

---

## Diagnostic Tools

### Run Full Analysis
```bash
python analyze_model_issue.py
```

Outputs:
- Dataset balance check
- Float32 vs INT8 confidence comparison
- Per-class performance breakdown
- Recommendations

### Output Example
```
[1] Dataset Analysis:
  Total samples: 250
  Classes: ['call', 'dislike', 'like', 'mute', 'ok']
  Class imbalance ratio: 1.00:1 (perfect)

[2] Model Quantization Analysis:
  Float32 mean confidence: 0.9044
  INT8 mean confidence: 0.3646
  Degradation: 59.7%

[4] Problem Diagnosis:
  ❌ LOW CONFIDENCE (0.3646) → Many predictions below 0.60 threshold
  ❌ HIGH QUANTIZATION LOSS (59.7%) → INT8 conversion damaged model
```

---

## See Also

- [MODEL_ISSUE_ANALYSIS.md](MODEL_ISSUE_ANALYSIS.md) — Detailed solutions & retraining guide

