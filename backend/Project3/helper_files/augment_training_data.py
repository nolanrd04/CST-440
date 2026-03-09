#!/usr/bin/env python3
"""
Augment training data to match captured camera characteristics.
Applies brightness, contrast, and gamma adjustments to match real camera output.
"""

import numpy as np
import cv2
import os
from pathlib import Path
import json

def apply_brightness_shift(img, shift):
    """Shift brightness by fixed amount."""
    return np.clip(img.astype(float) + shift, 0, 255).astype(np.uint8)

def apply_contrast_adjustment(img, scale):
    """Adjust contrast around midpoint."""
    mean = np.mean(img)
    return np.clip((img.astype(float) - mean) * scale + mean, 0, 255).astype(np.uint8)

def apply_gamma_correction(img, gamma):
    """Apply gamma correction curve."""
    img_norm = img.astype(float) / 255.0
    img_gamma = np.power(img_norm, gamma) * 255.0
    return np.clip(img_gamma, 0, 255).astype(np.uint8)

def apply_gaussian_blur(img, kernel_size=3):
    """Apply slight blur to reduce sharpness."""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def apply_inverse_vignetting(img, strength=0.1):
    """
    Apply inverse vignetting (brighten edges relative to center).
    This is unusual but matches the camera behavior.
    """
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]

    # Distance from center
    center_y, center_x = h // 2, w // 2
    radius = np.sqrt((Y - center_y)**2 + (X - center_x)**2)
    max_radius = np.sqrt(center_y**2 + center_x**2)

    # Normalize radius (0 at center, 1 at corners)
    radius_norm = radius / max_radius

    # Inverse vignetting: brighten edges
    vignette = 1.0 + strength * radius_norm

    img_float = img.astype(float)
    img_vignette = img_float * vignette
    return np.clip(img_vignette, 0, 255).astype(np.uint8)

def augment_image(img, brightness_shift=51, contrast_scale=0.76, gamma=1.3, blur=True, vignetting=False):
    """
    Apply all augmentations to single image.

    Args:
        img: Input image (0-255 range)
        brightness_shift: Amount to darken (positive = darker)
        contrast_scale: Scale factor (< 1 = lower contrast)
        gamma: Gamma value (> 1 = darker midtones)
        blur: Apply blur for sharpness reduction
        vignetting: Apply inverse vignetting
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Apply brightness shift (darken)
    img = apply_brightness_shift(img, -brightness_shift)

    # 2. Apply contrast reduction
    img = apply_contrast_adjustment(img, contrast_scale)

    # 3. Apply gamma correction (darken shadows/midtones)
    img = apply_gamma_correction(img, gamma)

    # 4. Apply slight blur to reduce sharpness
    if blur:
        img = apply_gaussian_blur(img, kernel_size=3)

    # 5. Apply inverse vignetting (brighten edges)
    if vignetting:
        img = apply_inverse_vignetting(img, strength=0.15)

    return img

def augment_dataset(X_train, y_train, X_val, y_val, X_test, y_test):
    """Augment full dataset to match camera characteristics."""

    print("\n" + "="*80)
    print("AUGMENTING TRAINING DATA TO MATCH CAMERA CHARACTERISTICS")
    print("="*80)

    print("\nAugmentation parameters:")
    print("  • Brightness shift: -51 levels (darken)")
    print("  • Contrast scale: 0.76x (reduce contrast by 24%)")
    print("  • Gamma: 1.3 (darker tonal curve)")
    print("  • Blur: Yes (reduce sharpness)")
    print("  • Vignetting: Yes (brighten edges)")

    # Augment training data
    print("\n📚 Augmenting training data...")
    X_train_aug = np.zeros_like(X_train)
    for i in range(len(X_train)):
        img = X_train[i, :, :, 0]  # (48, 48)
        img_uint8 = ((img * 0.224648 + 0.488811) * 255).astype(np.uint8)

        # Apply augmentations
        img_aug = augment_image(img_uint8,
                               brightness_shift=51,
                               contrast_scale=0.76,
                               gamma=1.3,
                               blur=True,
                               vignetting=True)

        # Renormalize to training range
        img_aug_norm = (img_aug.astype(float) / 255.0 - 0.488811) / 0.224648
        X_train_aug[i, :, :, 0] = img_aug_norm

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(X_train)} training samples")

    # Augment validation data
    print("\n📚 Augmenting validation data...")
    X_val_aug = np.zeros_like(X_val)
    for i in range(len(X_val)):
        img = X_val[i, :, :, 0]
        img_uint8 = ((img * 0.224648 + 0.488811) * 255).astype(np.uint8)
        img_aug = augment_image(img_uint8, brightness_shift=51, contrast_scale=0.76,
                               gamma=1.3, blur=True, vignetting=True)
        img_aug_norm = (img_aug.astype(float) / 255.0 - 0.488811) / 0.224648
        X_val_aug[i, :, :, 0] = img_aug_norm

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(X_val)} validation samples")

    # Augment test data
    print("\n📚 Augmenting test data...")
    X_test_aug = np.zeros_like(X_test)
    for i in range(len(X_test)):
        img = X_test[i, :, :, 0]
        img_uint8 = ((img * 0.224648 + 0.488811) * 255).astype(np.uint8)
        img_aug = augment_image(img_uint8, brightness_shift=51, contrast_scale=0.76,
                               gamma=1.3, blur=True, vignetting=True)
        img_aug_norm = (img_aug.astype(float) / 255.0 - 0.488811) / 0.224648
        X_test_aug[i, :, :, 0] = img_aug_norm

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(X_test)} test samples")

    return X_train_aug, X_val_aug, X_test_aug

def main():
    data_dir = Path('/home/sorenhaynes/Documents/GitHub/CST-440/backend/Project3/data/processed')

    if not data_dir.exists():
        print("❌ Data directory not found!")
        return

    # Load original data
    print("\n📂 Loading original training data...")
    X_train = np.load(data_dir / 'X_train.npy')
    y_train = np.load(data_dir / 'y_train.npy')
    X_val = np.load(data_dir / 'X_val.npy')
    y_val = np.load(data_dir / 'y_val.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_test = np.load(data_dir / 'y_test.npy')

    print(f"  ✓ X_train: {X_train.shape}")
    print(f"  ✓ X_val:   {X_val.shape}")
    print(f"  ✓ X_test:  {X_test.shape}")

    # Augment
    X_train_aug, X_val_aug, X_test_aug = augment_dataset(X_train, y_train, X_val, y_val, X_test, y_test)

    # Save augmented data with backup
    print("\n💾 Saving augmented data...")

    # Backup originals
    print("  Backing up original data...")
    np.save(data_dir / 'X_train_original.npy', X_train)
    np.save(data_dir / 'X_val_original.npy', X_val)
    np.save(data_dir / 'X_test_original.npy', X_test)

    # Save augmented
    np.save(data_dir / 'X_train.npy', X_train_aug)
    np.save(data_dir / 'X_val.npy', X_val_aug)
    np.save(data_dir / 'X_test.npy', X_test_aug)

    print("  ✓ Augmented training data saved")
    print("  ✓ Original data backed up as *_original.npy")

    print("\n✅ AUGMENTATION COMPLETE")
    print("\nNext steps:")
    print("  1. Run: python train_cnn_model.py")
    print("  2. Test the new model on your device")
    print("  3. If accuracy improves, the augmentation worked!")
    print("  4. If you want to revert: cp data/processed/*_original.npy data/processed/")

if __name__ == '__main__':
    main()
