#!/usr/bin/env python3
"""
Convert hex image data from capture_serial.py output to PNG image.
Usage: python hex_to_image.py capture_001.txt
"""
import sys
import re
import numpy as np
from PIL import Image

def hex_to_image(filename):
    """Read hex data from capture file and save as PNG."""
    with open(filename, 'r') as f:
        content = f.read()

    # Extract hex data between markers
    match = re.search(r'DEBUG_IMAGE_START(.*?)DEBUG_IMAGE_END', content, re.DOTALL)
    if not match:
        print(f"Error: Could not find DEBUG_IMAGE_START/END in {filename}")
        return False

    hex_section = match.group(1).strip()

    # Parse hex values (space or newline separated)
    hex_values = re.findall(r'[0-9a-fA-F]{2}', hex_section)

    if len(hex_values) != 48 * 48:
        print(f"Error: Expected {48*48} bytes, got {len(hex_values)}")
        return False

    # Convert hex to uint8 and reshape
    bytes_array = np.array([int(x, 16) for x in hex_values], dtype=np.uint8)
    img_array = bytes_array.reshape(48, 48)

    # Brighten the image for visibility (histogram equalization)
    from PIL import ImageEnhance
    img = Image.fromarray(img_array, 'L')

    # Apply brightness and contrast enhancement
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # Increase contrast
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.5)  # Increase brightness

    img_large = img.resize((384, 384), Image.NEAREST)  # 48*8 = 384

    output_filename = filename.replace('.txt', '.png')
    img_large.save(output_filename)

    print(f"✓ Saved {output_filename}")

    # Also print scores from the file
    scores = re.findall(r'(Non-face|Face) score:\s*([\d.]+)', content)
    if scores:
        print("\nScores:")
        for label, score in scores:
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
