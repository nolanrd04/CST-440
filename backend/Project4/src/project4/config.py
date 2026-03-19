from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GestureConfig:
    # Step 1: Problem definition and hardware-aware constraints.
    class_names: list[str] = field(default_factory=lambda: ["swipe_left", "swipe_right", "circle", "shake", "no_gesture"])
    sampling_hz: int = 100
    window_seconds: float = 2.0
    imu_channels: int = 6  # accel (x,y,z) + gyro (x,y,z)
    max_params_target: int = 50_000
    detection_threshold: float = 0.70
    cooldown_seconds: float = 1.0

    # Step 6 defaults.
    batch_size: int = 32
    epochs: int = 20
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # Paths.
    project_root: Path = Path(__file__).resolve().parents[2]
    artifacts_dir: Path = project_root / "artifacts"

    @property
    def window_size(self) -> int:
        return int(self.sampling_hz * self.window_seconds)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def ensure_dirs(cfg: GestureConfig) -> None:
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
