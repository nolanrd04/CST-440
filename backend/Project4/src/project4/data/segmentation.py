from __future__ import annotations

import numpy as np


def segment_stream(stream: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Step 3: Segment a continuous stream into fixed windows."""
    if stream.ndim != 2:
        raise ValueError("Expected stream shape [time, channels].")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive.")

    windows = []
    for start in range(0, max(stream.shape[0] - window_size + 1, 0), stride):
        windows.append(stream[start : start + window_size])

    if not windows:
        return np.empty((0, window_size, stream.shape[1]), dtype=stream.dtype)

    return np.stack(windows, axis=0)
