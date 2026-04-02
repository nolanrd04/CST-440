#!/usr/bin/env python3
"""
Convert hex image data from serial output to PNG image.
Usage: python hex_to_image.py capture_001.txt
Handles 96x96 grayscale images from Project 4.
"""
import sys
import re
import numpy as np
from PIL import Image

def hex_to_image(filename):
    """Read hex data from capture file and save as PNG."""
    with open(filename, 'r') as f:
        content = f.read()

    # Try to extract hex data between markers first
    match = re.search(r'DEBUG_IMAGE_START(.*?)DEBUG_IMAGE_END', content, re.DOTALL)
    if match:
        hex_section = match.group(1).strip()
    else:
        # If no markers, treat entire file as hex data
        hex_section = content

    # Parse hex values (space or newline separated)
    hex_values = re.findall(r'[0-9a-fA-F]{2}', hex_section)

    IMAGE_SIZE = 96
    expected_bytes = IMAGE_SIZE * IMAGE_SIZE

    if len(hex_values) > expected_bytes:
        print(f"Error: Too many bytes ({len(hex_values)}), expected at most {expected_bytes}")
        return False

    # Convert hex to uint8
    bytes_array = np.array([int(x, 16) for x in hex_values], dtype=np.uint8)

    # Pad with zeros if incomplete
    if len(bytes_array) < expected_bytes:
        print(f"Warning: Incomplete image. Got {len(bytes_array)} bytes, expected {expected_bytes}. Padding with zeros.")
        bytes_array = np.pad(bytes_array, (0, expected_bytes - len(bytes_array)), mode='constant', constant_values=0)

    img_array = bytes_array.reshape(IMAGE_SIZE, IMAGE_SIZE)

    # Brighten the image for visibility (histogram equalization)
    from PIL import ImageEnhance
    img = Image.fromarray(img_array, 'L')

    # Apply brightness and contrast enhancement
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # Increase contrast
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.5)  # Increase brightness

    img_large = img.resize((384, 384), Image.NEAREST)  # 96*4 = 384

    # Handle filename: replace extension or append .png
    if '.' in filename:
        output_filename = filename.rsplit('.', 1)[0] + '.png'
    else:
        output_filename = filename + '.png'
    img_large.save(output_filename)

    print(f"✓ Saved {output_filename}")

    # Also print scores from the file
    scores = re.findall(r'(\w+):\s*([\d.]+)', content)
    if scores:
        print("\nGesture Scores:")
        for label, score in scores:
            if label in ['call', 'dislike', 'like', 'mute', 'ok']:
                print(f"  {label}: {score}")

    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hex_to_image.py <capture_file.txt>")
        sys.exit(1)

    if hex_to_image(sys.argv[1]):
        sys.exit(0)
    else:
        sys.exit(1)
