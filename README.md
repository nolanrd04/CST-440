# CST-440 Project

## Quick Start Tutorial

### Step 1: Setup Environment

```sh
# Navigate to project
cd backend/Project1

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Train the Model

```sh
# Train the trigonometric neural network
python trig_model.py
```

This trains a model to compute sin, cos, and tan functions and saves it as `trig_model_all.keras`.

### Step 3: Convert to TensorFlow Lite

```sh
# Basic conversion (optimized)
python convert_to_tflite.py

# OR with quantization (recommended - 75% smaller!)
python convert_to_tflite.py --quantize
```

**Output files:**
- `trig_model_all.tflite` - Compressed model for Arduino
- `trig_model_all.h` - C header file with model as byte array

### Step 4: Deploy to Arduino

Copy `trig_model_all.h` to your Arduino sketch and include it:

```cpp
#include "trig_model_all.h"

void setup() {
  const tflite::Model* model = tflite::GetModel(trig_model);
  // Initialize interpreter and run inference...
}
```

---

## Understanding the Conversion

### What is TensorFlow Lite?

TFLite compresses TensorFlow models for embedded devices (Arduino has 1MB flash, 256KB RAM). It makes models smaller and faster.

### Conversion Steps

1. **Load Model** - Loads your trained `.keras` file (~0.5 MB)
2. **Optimize** - Reduces size by 30-50% (removes unnecessary ops, merges layers)
3. **Quantize** - Converts 32-bit floats → 8-bit integers (4x smaller, slight accuracy loss)
4. **Generate C Header** - Creates `.h` file with model as byte array for Arduino
5. **Test** - Verifies accuracy vs original model

### Quantization Explained

| Type | Size per Weight | Example (10K params) | Trade-off |
|------|----------------|---------------------|-----------|
| Float32 | 4 bytes | 40 KB | High precision |
| Int8 | 1 byte | 10 KB | 75% smaller, <0.01 accuracy loss |

Use `--quantize` for models >100KB or when you need faster inference.

### Command Options

```sh
python convert_to_tflite.py                        # Basic (optimized only)
python convert_to_tflite.py --quantize             # Quantized (recommended)
python convert_to_tflite.py --output path/         # Custom output dir
python convert_to_tflite.py --no-test --quiet      # Skip test, less output
```

### Model Architecture

**Input**: `[x_normalized, is_sin, is_cos, is_tan]` → **Output**: trig value

Example: `[0.5, 1, 0, 0]` computes sin(0)
