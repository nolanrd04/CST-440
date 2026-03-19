#!/usr/bin/env python3
"""
Analyze image characteristics from captured camera images vs training data.
Provides statistics and recommendations for augmenting training data.
"""

import numpy as np
import cv2
import os
from pathlib import Path
import json
from scipy import ndimage, signal

def calculate_laplacian_variance(image):
    """Calculate image sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def calculate_noise_level(image):
    """Estimate noise using Laplacian standard deviation."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    # Apply Laplacian and measure spread
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise = np.std(laplacian)
    return noise

def calculate_vignetting(image):
    """Detect vignetting by comparing edge vs center brightness."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape

    # Center region (middle 50%)
    cy1, cy2 = h//4, 3*h//4
    cx1, cx2 = w//4, 3*w//4
    center_brightness = np.mean(gray[cy1:cy2, cx1:cx2])

    # Edge regions (outer 25%)
    edge_brightness = (np.mean(gray[0:h//4, :]) +
                      np.mean(gray[3*h//4:h, :]) +
                      np.mean(gray[:, 0:w//4]) +
                      np.mean(gray[:, 3*w//4:w])) / 4

    vignetting = (center_brightness - edge_brightness) / center_brightness
    return vignetting

def analyze_gamma(image):
    """Estimate gamma curve by analyzing histogram shape."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()  # normalize

    # Gamma is estimated by comparing histogram to linear distribution
    # Lower gamma (darker) = more pixels in shadows
    # Higher gamma (brighter) = more pixels in highlights
    shadow_weight = np.sum(hist[:85])  # dark pixels (0-84)
    midtone_weight = np.sum(hist[85:170])  # midtones
    highlight_weight = np.sum(hist[170:])  # highlights

    return shadow_weight, midtone_weight, highlight_weight

def analyze_histogram(image):
    """Get detailed histogram statistics."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    return {
        'mean': np.mean(gray),
        'std': np.std(gray),
        'min': np.min(gray),
        'max': np.max(gray),
        'median': np.median(gray),
        'entropy': -np.sum((hist / hist.sum()) * np.log2(hist + 1e-10)),
        'histogram': hist.tolist()
    }

def analyze_image_set(image_paths, label=""):
    """Analyze a set of images."""
    results = {
        'brightness': [],
        'contrast': [],
        'sharpness': [],
        'noise': [],
        'vignetting': [],
        'gamma_shadow': [],
        'gamma_midtone': [],
        'gamma_highlight': [],
        'dynamic_range': [],
        'histograms': []
    }

    valid_images = 0

    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue

        try:
            # Read image as RGB
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ⚠ Failed to read {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)

            # Calculate metrics
            brightness = np.mean(gray)
            contrast = np.std(gray)
            sharpness = calculate_laplacian_variance(img)
            noise = calculate_noise_level(img)
            vignetting = calculate_vignetting(img)
            shadow, midtone, highlight = analyze_gamma(img)
            dynamic_range = np.max(gray) - np.min(gray)
            hist_stats = analyze_histogram(img)

            results['brightness'].append(brightness)
            results['contrast'].append(contrast)
            results['sharpness'].append(sharpness)
            results['noise'].append(noise)
            results['vignetting'].append(vignetting)
            results['gamma_shadow'].append(shadow)
            results['gamma_midtone'].append(midtone)
            results['gamma_highlight'].append(highlight)
            results['dynamic_range'].append(dynamic_range)
            results['histograms'].append(hist_stats)

            valid_images += 1
            print(f"  ✓ {Path(img_path).name}")
        except Exception as e:
            print(f"  ✗ {Path(img_path).name}: {e}")

    if valid_images == 0:
        print("  No valid images found!")
        return None

    # Compute statistics
    summary = {}
    for key in results:
        if key == 'histograms':
            continue
        if results[key]:
            summary[key] = {
                'mean': float(np.mean(results[key])),
                'std': float(np.std(results[key])),
                'min': float(np.min(results[key])),
                'max': float(np.max(results[key]))
            }

    print(f"\n  Analyzed {valid_images} images")
    return summary, results

def print_comparison(camera_stats, training_stats):
    """Print comparison between camera and training data."""
    print("\n" + "="*80)
    print("IMAGE CHARACTERISTICS COMPARISON")
    print("="*80)

    metrics = ['brightness', 'contrast', 'sharpness', 'noise', 'vignetting', 'dynamic_range']
    gamma_metrics = ['gamma_shadow', 'gamma_midtone', 'gamma_highlight']

    print("\n📊 PHOTOMETRIC PROPERTIES:")
    print("-" * 80)
    print(f"{'Metric':<20} {'Camera Mean':>15} {'Training Mean':>15} {'Difference':>15}")
    print("-" * 80)

    for metric in metrics:
        if metric in camera_stats and metric in training_stats:
            c_mean = camera_stats[metric]['mean']
            t_mean = training_stats[metric]['mean']
            diff = c_mean - t_mean
            diff_pct = (diff / t_mean * 100) if t_mean != 0 else 0

            print(f"{metric:<20} {c_mean:>15.2f} {t_mean:>15.2f} {diff_pct:>14.1f}%")

    print("\n📈 GAMMA/TONE DISTRIBUTION:")
    print("-" * 80)
    print(f"{'Component':<20} {'Camera':>15} {'Training':>15}")
    print("-" * 80)
    for metric in gamma_metrics:
        if metric in camera_stats and metric in training_stats:
            c_mean = camera_stats[metric]['mean']
            t_mean = training_stats[metric]['mean']
            print(f"{metric:<20} {c_mean:>15.3f} {t_mean:>15.3f}")

def generate_augmentation_recommendations(camera_stats, training_stats):
    """Generate specific augmentation recommendations."""
    print("\n" + "="*80)
    print("AUGMENTATION RECOMMENDATIONS")
    print("="*80)

    recommendations = []

    # Brightness adjustment
    brightness_diff = camera_stats['brightness']['mean'] - training_stats['brightness']['mean']
    if abs(brightness_diff) > 10:
        direction = "darker" if brightness_diff < 0 else "brighter"
        adjustment = int(brightness_diff)
        recommendations.append(
            f"1. BRIGHTNESS: Camera is {direction} by ~{abs(adjustment)} levels\n"
            f"   → Apply brightness shift of {-adjustment} to training data\n"
            f"   → Use: img = np.clip(img.astype(float) + {-adjustment}, 0, 255)"
        )

    # Contrast adjustment
    contrast_diff = camera_stats['contrast']['mean'] - training_stats['contrast']['mean']
    if abs(contrast_diff) > 5:
        if contrast_diff < 0:
            recommendations.append(
                f"2. CONTRAST: Camera has lower contrast ({contrast_diff:.1f} less)\n"
                f"   → Add noise or reduce contrast in training data\n"
                f"   → Use: blur filter or add Gaussian noise"
            )
        else:
            recommendations.append(
                f"2. CONTRAST: Camera has higher contrast ({contrast_diff:.1f} more)\n"
                f"   → Increase contrast in training data\n"
                f"   → Use: cv2.convertScaleAbs or CLAHE"
            )

    # Noise level
    noise_diff = camera_stats['noise']['mean'] - training_stats['noise']['mean']
    if noise_diff > 2:
        recommendations.append(
            f"3. SENSOR NOISE: Camera has significant noise ({noise_diff:.1f} higher)\n"
            f"   → Add Gaussian noise to training data\n"
            f"   → Use: img + np.random.normal(0, {noise_diff/2:.1f}, img.shape)"
        )

    # Vignetting
    vig_diff = camera_stats['vignetting']['mean']
    if abs(vig_diff) > 0.05:
        direction = "darkens" if vig_diff > 0 else "brightens"
        recommendations.append(
            f"4. VIGNETTING: Camera {direction} edges by ~{abs(vig_diff)*100:.1f}%\n"
            f"   → Add radial vignetting mask to training data\n"
            f"   → Create radial gradient mask and multiply with images"
        )

    # Dynamic range
    dr_diff = camera_stats['dynamic_range']['mean'] - training_stats['dynamic_range']['mean']
    if abs(dr_diff) > 20:
        direction = "narrower" if dr_diff < 0 else "wider"
        recommendations.append(
            f"5. DYNAMIC RANGE: Camera has {direction} range ({dr_diff:+.0f} levels)\n"
            f"   → Adjust levels/curves in training data"
        )

    # Gamma curve
    shadow_diff = camera_stats['gamma_shadow']['mean'] - training_stats['gamma_shadow']['mean']
    if abs(shadow_diff) > 0.05:
        recommendations.append(
            f"6. GAMMA CURVE: Camera shows different tonal distribution\n"
            f"   → Apply gamma correction: img^(1/gamma) or img^gamma\n"
            f"   → Estimate gamma from histogram shape"
        )

    if not recommendations:
        recommendations.append("No significant differences detected - training data matches camera well!")

    for rec in recommendations:
        print(f"\n{rec}")

def main():
    print("\n" + "="*80)
    print("CAMERA IMAGE vs TRAINING DATA ANALYSIS")
    print("="*80)

    # Find captured images
    capture_dir = Path('/home/sorenhaynes/Documents/GitHub/CST-440/backend/Project3')
    capture_images = sorted(capture_dir.glob('capture_*.png'))

    if not capture_images:
        print("❌ No captured images found!")
        return

    print(f"\n📷 CAPTURED IMAGES ({len(capture_images)} found):")
    camera_stats, camera_results = analyze_image_set(capture_images, "Camera")

    # Find training data
    train_data_dir = Path('/home/sorenhaynes/Documents/GitHub/CST-440/backend/Project3/data/processed')

    if not train_data_dir.exists():
        print("\n❌ Training data directory not found!")
        return

    # Load training images (X_train.npy)
    X_train_path = train_data_dir / 'X_train.npy'
    if X_train_path.exists():
        print(f"\n📚 TRAINING DATA (from X_train.npy):")
        X_train = np.load(X_train_path)  # Should be (N, 48, 48, 1) and normalized

        # Denormalize to 0-255 for fair comparison
        # Assuming normalization: (img / 255 - mean) / std
        # Reverse: img = (normalized * std + mean) * 255
        PIXEL_MEAN = 0.488811
        PIXEL_STD = 0.224648

        training_images_denorm = []
        for i in range(min(10, len(X_train))):
            img = X_train[i, :, :, 0]  # (48, 48)
            # Denormalize
            img_denorm = (img * PIXEL_STD + PIXEL_MEAN) * 255
            img_denorm = np.clip(img_denorm, 0, 255)
            training_images_denorm.append(img_denorm)

        # Analyze denormalized training data
        print("  Analyzing denormalized training samples...")
        train_stats = {
            'brightness': [],
            'contrast': [],
            'sharpness': [],
            'noise': [],
            'vignetting': [],
            'gamma_shadow': [],
            'gamma_midtone': [],
            'gamma_highlight': [],
            'dynamic_range': []
        }

        for img in training_images_denorm:
            img_uint8 = img.astype(np.uint8)
            train_stats['brightness'].append(np.mean(img))
            train_stats['contrast'].append(np.std(img))
            train_stats['sharpness'].append(cv2.Laplacian(img_uint8, cv2.CV_64F).var())
            train_stats['noise'].append(np.std(cv2.Laplacian(img_uint8, cv2.CV_64F)))
            train_stats['vignetting'].append(0)  # synthetic data typically has no vignetting
            # Convert to RGB for analyze_gamma
            img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
            s, m, h = analyze_gamma(img_rgb)
            train_stats['gamma_shadow'].append(s)
            train_stats['gamma_midtone'].append(m)
            train_stats['gamma_highlight'].append(h)
            train_stats['dynamic_range'].append(np.max(img) - np.min(img))

        # Convert to summary format
        training_summary = {}
        for key in train_stats:
            if train_stats[key]:
                training_summary[key] = {
                    'mean': float(np.mean(train_stats[key])),
                    'std': float(np.std(train_stats[key])),
                    'min': float(np.min(train_stats[key])),
                    'max': float(np.max(train_stats[key]))
                }

        print(f"  ✓ Analyzed {len(training_images_denorm)} training samples")

        # Compare
        print_comparison(camera_stats, training_summary)
        generate_augmentation_recommendations(camera_stats, training_summary)

        # Save detailed report
        report = {
            'camera': camera_stats,
            'training': training_summary,
            'timestamp': str(Path.cwd())
        }
        report_path = capture_dir / 'image_analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Report saved to: {report_path}")
    else:
        print(f"\n❌ Training data not found at {X_train_path}")

if __name__ == '__main__':
    main()
