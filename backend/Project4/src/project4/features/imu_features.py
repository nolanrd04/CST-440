from __future__ import annotations

import numpy as np


def _rms(x: np.ndarray, axis: int) -> np.ndarray:
    return np.sqrt(np.mean(np.square(x), axis=axis))


def _fft_energy(x: np.ndarray, axis: int) -> np.ndarray:
    freq = np.fft.rfft(x, axis=axis)
    return np.mean(np.abs(freq) ** 2, axis=axis)


def extract_imu_features(windows: np.ndarray) -> np.ndarray:
    """Step 4a: Statistical + frequency features from IMU windows."""
    if windows.ndim != 3:
        raise ValueError("Expected windows shape [batch, time, channels].")

    means = windows.mean(axis=1)
    variances = windows.var(axis=1)
    rms = _rms(windows, axis=1)
    fft_energy = _fft_energy(windows, axis=1)

    # Signal magnitude vector from first 3 channels (accelerometer).
    smv = np.linalg.norm(windows[:, :, :3], axis=2).mean(axis=1, keepdims=True)

    features = np.concatenate([means, variances, rms, fft_energy, smv], axis=1)

    # Normalize features per-dimension.
    mu = features.mean(axis=0, keepdims=True)
    sigma = features.std(axis=0, keepdims=True) + 1e-6
    return ((features - mu) / sigma).astype(np.float32)
