#!/bin/bash

# Face Detector - Arduino Upload Script
# Compiles and uploads the sketch to your Arduino

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR"
SKETCH_NAME="face_detector"

echo "=========================================="
echo "Face Detector - Arduino Upload"
echo "=========================================="
echo ""

# Check if arduino-cli is installed
if ! command -v arduino-cli &> /dev/null; then
    echo "❌ arduino-cli not found"
    exit 1
fi

# Find available ports
echo "Available ports:"
PORTS=$(arduino-cli board list | grep -v "Port\|---" | awk '{print $1}')

if [ -z "$PORTS" ]; then
    echo "❌ No Arduino boards found. Check USB connection."
    exit 1
fi

echo "$PORTS"
echo ""

# Let user select port if multiple
PORTS_ARRAY=($PORTS)
if [ ${#PORTS_ARRAY[@]} -eq 1 ]; then
    PORT=${PORTS_ARRAY[0]}
    echo "Using port: $PORT"
else
    echo "Multiple ports found. Select one:"
    PS3="Enter choice: "
    select port in "${PORTS_ARRAY[@]}"; do
        PORT="$port"
        break
    done
fi

echo ""

# Detect board from connected device
echo "Detecting board type..."
BOARD=$(arduino-cli board list | grep "$PORT" | awk '{print $2}')

if [ -z "$BOARD" ]; then
    echo "Could not detect board. Please specify manually:"
    echo "  arduino:samd:nano_33_iot"
    echo "  arduino:mbed_nano:nano33ble"
    read -p "Enter board: " BOARD
fi

echo "Board detected: $BOARD"
echo ""

# Compile
echo "Compiling sketch..."
arduino-cli compile -b "$BOARD" "$PROJECT_DIR" --verbose

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Compilation failed"
    exit 1
fi

echo ""
echo "✓ Compilation successful"
echo ""

# Upload
echo "Uploading to $PORT..."
arduino-cli upload -p "$PORT" -b "$BOARD" "$PROJECT_DIR" --verbose

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Upload failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Upload successful!"
echo ""
echo "Opening serial monitor..."
echo "Press Ctrl+C to exit"
echo "=========================================="
echo ""

# Try to open serial monitor
if command -v screen &> /dev/null; then
    screen "$PORT" 115200
elif command -v minicom &> /dev/null; then
    minicom -D "$PORT" -b 115200
else
    echo "Install 'screen' or 'minicom' to monitor serial output:"
    echo "  sudo apt install screen"
    echo ""
    echo "Then run: screen $PORT 115200"
fi
