#!/bin/bash

# Simple Arduino CLI build and upload script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Arduino Nano 33 BLE
BOARD="arduino:mbed_nano:nano33ble"

echo "=========================================="
echo "Face Detector - Arduino Build & Upload"
echo "=========================================="
echo ""

# List available ports
echo "Available ports:"
arduino-cli board list | grep -v "Port\|---"
echo ""

read -p "Enter port (e.g., /dev/ttyACM0): " PORT

if [ -z "$PORT" ]; then
    echo "Error: Port required"
    exit 1
fi

echo ""
echo "Building for board: $BOARD"
echo "Port: $PORT"
echo ""

# Compile
echo "Compiling..."
arduino-cli compile -b "$BOARD" "$SCRIPT_DIR"

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
arduino-cli upload -p "$PORT" -b "$BOARD" "$SCRIPT_DIR"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Upload failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Upload complete!"
echo "=========================================="
echo ""
echo "Open serial monitor at 115200 baud:"
echo "  screen $PORT 115200"
echo "  (or use your favorite serial monitor)"
echo ""
echo "In the monitor, type 'c' to capture and infer."
