#!/bin/bash

# Face Detector - Arduino Setup Script
# Installs dependencies and configures the project for Arduino CLI

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR"

echo "=========================================="
echo "Face Detector - Arduino Setup"
echo "=========================================="
echo ""

# Check if arduino-cli is installed
if ! command -v arduino-cli &> /dev/null; then
    echo "❌ arduino-cli not found. Please install it first:"
    echo "   https://arduino.github.io/arduino-cli/latest/installation/"
    exit 1
fi

echo "✓ arduino-cli found"
echo ""

# Update board index
echo "Updating board index..."
arduino-cli core update-index

# Install Arduino Nano 33 board core (choose one)
echo ""
echo "Select your Arduino Nano 33 variant:"
echo "  1) Arduino Nano 33 IoT (with WiFi)"
echo "  2) Arduino Nano 33 BLE (with Bluetooth)"
echo ""
read -p "Enter choice (1 or 2): " choice

case $choice in
  1)
    BOARD="arduino:samd:nano_33_iot"
    echo "Installing Arduino SAMD boards..."
    arduino-cli core install arduino:samd
    ;;
  2)
    BOARD="arduino:mbed_nano:nano33ble"
    echo "Installing Arduino nRF52 boards..."
    arduino-cli core install arduino:mbed_nano
    ;;
  *)
    echo "Invalid choice"
    exit 1
    ;;
esac

echo ""
echo "Installing required libraries..."
arduino-cli lib install "TensorFlow Lite for Microcontrollers"
arduino-cli lib install "ArduCAM"

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "Board selected: $BOARD"
echo ""
echo "To upload the sketch, run:"
echo "  ./upload.sh"
echo "=========================================="
