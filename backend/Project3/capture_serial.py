#!/usr/bin/env python3
"""
Capture serial output from face detector Arduino and save to file.
Useful for debugging camera image data and inference results.

Usage:
    python3 capture_serial.py              # Uses /dev/ttyACM0 (default)
    python3 capture_serial.py /dev/ttyUSB0 # Uses specified port
"""
import serial
import time
import sys

def main():
    # Get port from command line or use default
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = '/dev/ttyACM0'

    baud = 115200

    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException:
        print(f"Error: Could not open {port}")
        print("Available ports:")
        import glob
        ports = glob.glob('/dev/tty*')
        for p in ports:
            print(f"  {p}")
        sys.exit(1)

    time.sleep(2)  # Wait for Arduino to initialize

    print(f"Connected to {port} at {baud} baud")
    print("Waiting for captures... (send 'c' to the Arduino to capture)")
    print("Press Ctrl+C to exit\n")

    capture_number = 0
    output = []

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(line)
                output.append(line)

                # Save when we get the end marker
                if 'DEBUG_IMAGE_END' in line:
                    capture_number += 1
                    filename = f'capture_{capture_number:03d}.txt'
                    with open(filename, 'w') as f:
                        f.write('\n'.join(output))
                    print(f"\n✓ Saved to {filename}\n")
                    output = []
    except KeyboardInterrupt:
        ser.close()
        print("\n\nDone!")
        sys.exit(0)

if __name__ == "__main__":
    main()
