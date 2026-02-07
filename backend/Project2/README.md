# Keyword Spotting Project

This project implements a keyword spotting system using machine learning models. It includes scripts for data preprocessing, training, and exporting models for deployment on Arduino devices.

## Project Structure

- `DataImporter.py`: Script to import and organize raw audio data.
- `DataPreprocessor.py`: Script to preprocess audio data and generate training, validation, and test datasets.
- `train_gru_model.py`: Script to train a GRU-based model for keyword spotting.
- `kws_model.keras`: Trained model in Keras format.
- `kws_model.tflite`: Trained model in TensorFlow Lite format for deployment.
- `data/`: Contains raw and processed audio data.
  - `archive/`: Raw audio data organized by keyword.
  - `processed/`: Preprocessed data and metadata files.
- `keyword_spotting_arduino/`: Contains Arduino deployment files.
  - `keyword_spotting_arduino.ino`: Arduino sketch for keyword spotting.
  - `kws_model_data.h`: Model data for Arduino.
- `Report.ipynb`: Jupyter notebook for generating reports and visualizations.
- `requirements.txt`: Python dependencies for the project.

## Setup Instructions

### 1. Install Dependencies

Ensure you have Python installed. Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data

Run the `DataImporter.py` and `DataPreprocessor.py` scripts to prepare the data:

```bash
python DataImporter.py
python DataPreprocessor.py
```

### 3. Train the Model

Train the GRU-based model using the `train_gru_model.py` script:

```bash
python train_gru_model.py
```

This will generate the `kws_model.keras` and `kws_model.tflite` files.

### 4. Test the Model

Use the `Report.ipynb` notebook to evaluate the model's performance. Open the notebook in Jupyter or VS Code and run the cells.

### 5. Deploy to Arduino

#### Option 1: Using Arduino IDE

**Step 1: Install Board Support**
1. Open Arduino IDE
2. Go to **Tools > Board > Boards Manager**
3. Search for "Arduino Mbed OS Nano Boards"
4. Install "Arduino Mbed OS Nano Boards" by Arduino

**Step 2: Install Required Libraries**
1. Go to **Tools > Manage Libraries**
2. Search and install these libraries:
   - **PDM** (by Arduino) - for microphone input
   - **CMSIS-DSP** (or "CMSIS") - for signal processing

3. **Chirale_TensorFlowLite** - Install manually:
   - Download: https://github.com/ChiraleBrandon/Chirale_TensorFlowLite
   - Go to **Sketch > Include Library > Add .ZIP Library**
   - Select the downloaded ZIP file

**Step 3: Connect and Configure Board**
1. Connect Arduino Nano 33 BLE via USB
2. Select board: **Tools > Board > Arduino Nano 33 BLE**
3. Select port: **Tools > Port** 
   - Mac: `/dev/cu.usbmodem14201` or similar
   - Windows: `COM3` or similar
   - Linux: `/dev/ttyACM0` or similar

**Step 4: Upload Code**
1. Open `keyword_spotting_arduino/keyword_spotting_arduino.ino`
2. Click **Upload** button (→)
   - First upload takes 2-5 minutes due to library size
   - Subsequent uploads are faster
3. Wait for "Done uploading" message

**Step 5: Test the System**
1. Open **Tools > Serial Monitor** (set to 115200 baud)
2. You should see initialization messages
3. **Say the wake word**: "Sheila"
   - System enters listening mode for 25 seconds
   - Serial monitor shows: `[WAKE] 'sheila' detected! Listening for commands...`
4. **Say a keyword**: "down", "off", "on", "up", or "wow"
   - System outputs `1` for 1 second when keyword detected
   - Serial monitor shows: `[KEYWORD] Detected: [word] (confidence: X.XX)`
5. System returns to waiting mode after 25 seconds of no commands

**Expected Serial Output:**
```
========================================
Keyword Spotting with Wake Word
CST-440 Project 2
========================================
Signal processing initialized.
Model loaded. Input shape: 1x49x26, Output shape: 1x8
Microphone started.
0
0
[WAKE] 'sheila' detected! Listening for commands...
0
[KEYWORD] Detected: on (confidence: 0.85)
1
0
[TIMEOUT] Returning to WAITING state.
0
```

#### Option 2: Using PlatformIO in VS Code

1. Install the [PlatformIO extension for VS Code](https://marketplace.visualstudio.com/items?itemName=platformio.platformio-ide)
2. Open the `keyword_spotting_arduino/` folder in VS Code
3. Connect Arduino Nano 33 BLE via USB
4. Open `platformio.ini` and verify settings:
   - `board = nano33ble`
   - `upload_port = /dev/cu.usbmodemXXXX` (update with your port)
5. Click **Upload** button (→) in PlatformIO toolbar
6. Open **Serial Monitor** to view output

---

## Testing Voice Commands

**System Behavior:**
- **Waiting Mode**: Listens only for "Sheila" (wake word) → Outputs `0`
- **Listening Mode**: Active for 25 seconds after wake word detected
  - Recognizes: "down", "off", "on", "up", "wow"
  - Outputs `1` for 1 second when keyword detected
  - Returns to waiting mode after timeout

**Tips:**
- Speak clearly at normal volume
- Hold microphone 10-20cm from your mouth
- Minimize background noise for better accuracy
- System shows confidence scores in serial monitor

---

## Troubleshooting

**Error: "No device found on cu.usbmodemXXXX"**
- Reconnect USB cable
- Try different USB port
- Double-tap RESET button on Arduino (enters bootloader mode)
- Check port in Tools > Port

**Error: "Type INT32 not supported" or "Invoke() failed"**
- Model needs to be regenerated with fixed training script
- Run: `python train_gru_model.py`
- This generates new `kws_model.tflite` and `kws_model_data.h`
- Re-upload Arduino code

**Low accuracy / Not detecting keywords:**
- Increase volume when speaking
- Reduce background noise
- Check Serial Monitor for confidence scores (should be > 0.6)
- Ensure microphone is facing you

---

## Notes

- Arduino Nano 33 BLE has 1MB flash and 256KB RAM
- Model uses ~130KB flash, ~100KB RAM
- Background noise samples in `data/_background_noise_/` improve robustness
- See `data/archive/README.md` for dataset details
- Confidence threshold set to 0.6 (adjustable in Arduino code)