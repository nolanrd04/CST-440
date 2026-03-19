from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from project4.config import GestureConfig


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


class SyntheticIMUDatasetAdapter:
    """Step 2 placeholder adapter until real dataset is selected."""

    def __init__(self, cfg: GestureConfig, samples_per_class: int = 400) -> None:
        self.cfg = cfg
        self.samples_per_class = samples_per_class

    def load_windows(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.cfg.random_seed)
        num_classes = self.cfg.num_classes
        windows_per_class = self.samples_per_class
        t = np.linspace(0.0, self.cfg.window_seconds, self.cfg.window_size, endpoint=False)

        x, y = [], []
        for class_idx in range(num_classes):
            for _ in range(windows_per_class):
                base = rng.normal(0.0, 0.2, size=(self.cfg.window_size, self.cfg.imu_channels))
                freq = 0.5 + class_idx * 0.35
                signal = np.sin(2 * np.pi * freq * t)
                base[:, 0] += signal
                base[:, 1] += np.cos(2 * np.pi * freq * t)
                base[:, 2] += 0.5 * signal * signal
                x.append(base.astype(np.float32))
                y.append(class_idx)

        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)


class RealIMUDatasetAdapter:
    """Implement this once train/validation/test dataset paths are finalized."""

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg

    def load_raw_streams(self) -> tuple[list[np.ndarray], list[int]]:
        raise NotImplementedError("Connect real IMU streams and labels here.")


def split_dataset(x: np.ndarray, y: np.ndarray, cfg: GestureConfig) -> SplitData:
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x,
        y,
        test_size=cfg.val_ratio + cfg.test_ratio,
        random_state=cfg.random_seed,
        stratify=y,
    )

    val_part = cfg.val_ratio / (cfg.val_ratio + cfg.test_ratio)
    x_val, x_test, y_val, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=1.0 - val_part,
        random_state=cfg.random_seed,
        stratify=y_holdout,
    )

    return SplitData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )
