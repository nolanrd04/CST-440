# Keyword Spotting on Arduino Nano 33 BLE

This project implements a keyword spotting system with wake word detection on Arduino Nano 33 BLE using TensorFlow Lite for Microcontrollers.

## Hardware Requirements
- Arduino Nano 33 BLE board
- Built-in PDM microphone (on board)
- USB cable for programming

## Software Requirements
- Arduino IDE 2.x (recommended) or Arduino IDE 1.8.x

## Setup Instructions (Arduino IDE)

### 1. Install Arduino IDE
Download and install from: https://www.arduino.cc/en/software

### 2. Install Board Support
1. Open Arduino IDE
2. Go to **Tools > Board > Boards Manager**
3. Search for "Arduino Mbed OS Nano Boards"
4. Install "Arduino Mbed OS Nano Boards" by Arduino

### 3. Install Required Libraries
1. Go to **Tools > Manage Libraries** (or Sketch > Include Library > Manage Libraries)
2. Search and install the following libraries:
   - **PDM** (by Arduino) - for microphone input
   - **Chirale_TensorFlowLite** (by Chirale) - for TensorFlow Lite inference
     - If not found in Library Manager, manually install:
       1. Download from: https://github.com/ChiraleBrandon/Chirale_TensorFlowLite
       2. Go to Sketch > Include Library > Add .ZIP Library
       3. Select the downloaded ZIP file

### 4. Configure Board Settings
1. Connect your Arduino Nano 33 BLE to your computer via USB
2. Go to **Tools > Board** and select **Arduino Nano 33 BLE**
3. Go to **Tools > Port** and select the port showing your Arduino
   - On Mac: looks like `/dev/cu.usbmodem14201` or similar
   - On Windows: looks like `COM3` or similar
   - On Linux: looks like `/dev/ttyACM0` or similar

### 5. Open the Project
1. Open `keyword_spotting_arduino.ino` in Arduino IDE
2. The IDE should automatically open `kws_model_data.h` as a tab

### 6. Compile and Upload
1. Click the **Verify** button (✓) to compile the code
   - First compilation will take 2-5 minutes due to TensorFlow Lite library size
   - Subsequent compilations will be faster
2. Click the **Upload** button (→) to upload to the board
3. Open **Tools > Serial Monitor** to see debug output (set to 115200 baud)

## How It Works

### State Machine
- **WAITING mode**: Only responds to "sheila" (wake word)
- **LISTENING mode**: Detects keywords (down, off, on, up, wow)
- After 25 seconds of inactivity in LISTENING mode, returns to WAITING

### Audio Processing
- Microphone: PDM at 16kHz sample rate
- Buffer: 1 second of audio
- Features: 13 MFCCs + 13 delta MFCCs = 26 features per frame
- Frames: 49 frames (30ms window, 20ms stride)

### Model
- Architecture: GRU(48) → GRU(48) → 8-class softmax
- Format: TensorFlow Lite (float32)
- Size: ~128KB

### Output
- Outputs `0` in WAITING mode
- Outputs `1` when a keyword is detected in LISTENING mode

## Troubleshooting

### Board Not Found
- Make sure the USB cable supports data transfer (not just charging)
- Try a different USB port
- Press the reset button twice quickly to enter bootloader mode

### Compilation Errors
- Ensure all libraries are installed correctly
- Close and reopen Arduino IDE
- Try clearing the Arduino cache: Delete `~/Library/Arduino15/` (Mac) or `%APPDATA%/Arduino15/` (Windows)

### Upload Fails
- Press reset button twice quickly before uploading
- Try a different USB cable
- Check that the correct port is selected

### No Audio Detection
- Speak clearly and close to the microphone (on the board)
- Ensure Serial Monitor is set to 115200 baud to see debug output
- Check that the built-in LED blinks when processing audio

## Alternative: PlatformIO (Advanced Users)

If you prefer command-line tools:

```bash
# Install PlatformIO
pip install platformio

# Build the project
platformio run

# Upload to board
platformio run --target upload

# Monitor serial output
platformio device monitor
```

Note: PlatformIO initial setup can be slow due to large downloads. Arduino IDE is recommended for simpler workflow.
