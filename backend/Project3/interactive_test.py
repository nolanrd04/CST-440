#!/usr/bin/env python3
"""
Interactive face detection test - press Enter to capture
Saves captures as capture_NNN.txt and capture_NNN.png
"""
import serial
import time
import sys
import re
import glob
import numpy as np
from PIL import Image, ImageEnhance

PORT = '/dev/cu.usbmodem1101'
BAUD = 115200

def get_next_capture_number():
    """Find the next available capture number."""
    existing_files = glob.glob('capture_*.txt')
    if not existing_files:
        return 1
    
    numbers = []
    for f in existing_files:
        match = re.search(r'capture_(\d+)\.txt', f)
        if match:
            numbers.append(int(match.group(1)))
    
    return max(numbers) + 1 if numbers else 1

def save_capture(output_lines, capture_num):
    """Save capture output as both TXT and PNG."""
    # Save text file
    txt_filename = f'capture_{capture_num:03d}.txt'
    with open(txt_filename, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"✓ Saved {txt_filename}")
    
    # Extract and save image
    content = '\n'.join(output_lines)
    match = re.search(r'DEBUG_IMAGE_START(.*?)DEBUG_IMAGE_END', content, re.DOTALL)
    
    if not match:
        print("  (No debug image found in output)")
        return
    
    hex_section = match.group(1).strip()
    hex_values = re.findall(r'[0-9a-fA-F]{2}', hex_section)
    
    if len(hex_values) != 48 * 48:
        print(f"  Warning: Expected {48*48} hex bytes, got {len(hex_values)}")
        return
    
    # Convert hex to image
    bytes_array = np.array([int(x, 16) for x in hex_values], dtype=np.uint8)
    img_array = bytes_array.reshape(48, 48)
    img = Image.fromarray(img_array, 'L')
    
    # Enhance visibility
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.5)
    
    # Resize for easier viewing
    img_large = img.resize((384, 384), Image.NEAREST)
    
    png_filename = f'capture_{capture_num:03d}.png'
    img_large.save(png_filename)
    print(f"✓ Saved {png_filename}")


try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(0.5)
    print(f"✓ Connected to {PORT}")
    
    # Read initialization messages
    print("\nInitialization:")
    for _ in range(10):
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"  {line}")
        time.sleep(0.1)
    
    print("\n" + "="*60)
    print("Ready! Press Enter to capture, type 'q' to quit")
    print("="*60 + "\n")
    
    while True:
        cmd = input(">>> Press Enter to capture (q to quit): ").strip().lower()
        
        if cmd == 'q':
            break
        
        # Send 'c' command
        print("\n📸 Sending capture command...")
        ser.write(b'c')
        time.sleep(0.1)
        
        # Read response (max 10 seconds)
        print("\n--- Arduino Output ---\n")
        start_time = time.time()
        found_result = False
        output_lines = []  # Store all output for saving
        
        while time.time() - start_time < 10:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(line)
                    output_lines.append(line)
                    if 'DETECTED' in line or 'NO FACE' in line:
                        found_result = True
            
            if found_result and ser.in_waiting == 0:
                break
            
            time.sleep(0.01)
        
        # Save capture files
        if output_lines:
            print("\n--- Saving Capture ---\n")
            capture_num = get_next_capture_number()
            save_capture(output_lines, capture_num)
        
        print("\n" + "="*60 + "\n")
    
    ser.close()
    print("\nDisconnected")

except serial.SerialException as e:
    print(f"❌ Serial port error: {e}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n\nInterrupted, disconnected")
    if 'ser' in locals():
        ser.close()
