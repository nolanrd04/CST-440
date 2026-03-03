# Face Detector - Arduino Nano 33 + Arducam Mini OV2640

TensorFlow Lite face detection model deployed on Arduino with camera input.

## Files

- `face_detector.ino` - Main sketch
- `face_model_data.h` - Embedded TFLite model
- `setup.sh` - Install dependencies
- `upload.sh` - Compile and upload
- `arduino-cli.yaml` - Arduino CLI configuration

## Requirements

- Arduino Nano 33 (IoT or BLE)
- Arducam Mini OV2640 module
- Arduino CLI (install from: https://arduino.github.io/arduino-cli/latest/installation/)
- USB cable

## Wiring

| Camera Pin | Arduino Nano 33 |
|-----------|-----------------|
| MOSI | D11 |
| MISO | D12 |
| SCK | D13 |
| CS | D10 |
| SDA | A4 |
| SCL | A5 |
| GND | GND |
| VCC | 3.3V |

## Setup & Upload

### 1. Install dependencies
```bash
cd arduino_sketch
./setup.sh
```

Follow prompts to:
- Select your Arduino Nano 33 variant (IoT or BLE)
- Install board cores and libraries

### 2. Upload sketch
```bash
./upload.sh
```

This will:
- Detect your Arduino board and port
- Compile the sketch
- Upload to the Arduino
- Open serial monitor (115200 baud)

### 3. Test

In the serial monitor:
1. Type `c` and press Enter
2. Watch the output:
   ```
   Capturing frame...
   Frame captured. Reading data...
   Processing image...
   Running inference...

   Non-face score: 0.1234
   Face score:     0.8766

   ✓ FACE DETECTED
   ```

## Manual Setup (if scripts don't work)

```bash
# Install board core
arduino-cli core install arduino:samd

# Install libraries
arduino-cli lib install "TensorFlow Lite for Microcontrollers"
arduino-cli lib install "ArduCAM"

# Compile
arduino-cli compile -b arduino:samd:nano_33_iot arduino_sketch

# Upload (replace PORT and BOARD as needed)
arduino-cli upload -p /dev/ttyACM0 -b arduino:samd:nano_33_iot arduino_sketch

# Monitor
screen /dev/ttyACM0 115200
```

## Troubleshooting

**Board not detected:**
- Check USB connection
- Try `arduino-cli board list` to see available ports
- Check Device Manager (Windows) or `ls /dev/tty*` (Linux)

**Compilation error - undefined reference:**
- Ensure libraries are installed: `arduino-cli lib list`
- Delete build cache: `rm -rf ~/.arduino15`

**Camera not initializing:**
- Verify SPI/I2C wiring
- Check camera power supply (3.3V, not 5V)
- Try `setup.sh` again

**Memory issues:**
- The tensor arena is 80KB, should fit on Nano 33
- Monitor with serial debug output

## Camera Output

The sketch captures 320×240 RGB565 frames, resizes to 48×48 grayscale, normalizes, and runs inference. Outputs:
- Non-face confidence score [0, 1]
- Face confidence score [0, 1]
- Classification result

## Next Steps

- Optimize camera resolution for better accuracy
- Add JPEG decoding for faster captures
- Implement continuous detection loop
- Add LED/buzzer for alerts on face detection
