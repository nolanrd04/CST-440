# Face Detector Debug Tools

Tools for capturing and visualizing camera data from the Arduino face detector.

## Setup

```bash
pip install pyserial pillow numpy
```

## Workflow

### 1. Capture Serial Output

Run the capture script in one terminal:

```bash
python capture_serial.py
```

This will:
- Connect to `/dev/ttyACM0` at 115200 baud
- Wait for captures from the Arduino
- Automatically save each capture to `capture_001.txt`, `capture_002.txt`, etc.

### 2. Send Capture Command

In the Arduino serial monitor (or another terminal), send `c` to trigger a capture:

```bash
echo "c" > /dev/ttyACM0
```

Or use `pio device monitor` and type `c`.

### 3. Convert to Image

Once you have a capture file, convert the hex image data to PNG:

```bash
python hex_to_image.py capture_001.txt
```

This will:
- Extract the 48×48 hex image data
- Save as `capture_001.png` (enlarged 8× for visibility)
- Print the inference scores

## What to Look For

The captured images show:
- **What the camera actually sees** after preprocessing
- **Grayscale 0-255 values** from BT.601 luminance conversion
- **After nearest-neighbor resize** from 320×240 → 48×48

**Debug checklist:**
- [ ] Face image: does it show a recognizable face?
- [ ] Wall image: is it uniform/blank?
- [ ] Are the grayscale values reasonable (not all black, not all white)?
- [ ] Is RGB565 decoding correct (face/wall have different patterns)?

## Example

```bash
# Terminal 1: Capture
python capture_serial.py

# Terminal 2: Send captures
echo "c" > /dev/ttyACM0
echo "c" > /dev/ttyACM0

# Back to Terminal 1: Should have capture_001.txt and capture_002.txt

# Terminal 3: Convert to images
python hex_to_image.py capture_001.txt
python hex_to_image.py capture_002.txt
open capture_001.png
open capture_002.png
```

## Troubleshooting

**"Could not open /dev/ttyACM0"**
- Check Arduino is connected: `ls /dev/tty*`
- Update the port in `capture_serial.py`

**"Expected 2304 bytes, got X"**
- The serial output was truncated
- Make sure the full `DEBUG_IMAGE_END` marker is in the file

**Images look wrong**
- Check if grayscale conversion is correct
- Verify RGB565 decoding in the Arduino sketch
- Check if the camera is actually capturing (compare with dummy black image)
