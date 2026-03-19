from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class RuntimeState(str, Enum):
    WAITING = "waiting"
    DETECTING = "detecting"
    COOLDOWN = "cooldown"


@dataclass
class GestureStateMachine:
    threshold: float
    cooldown_seconds: float

    state: RuntimeState = RuntimeState.WAITING
    last_trigger_time: float = -1e9
    last_class: int | None = None

    def step(self, probs: np.ndarray, now_s: float) -> int | None:
        """Step 10: threshold + cooldown + anti-repeat logic."""
        cls = int(np.argmax(probs))
        conf = float(probs[cls])

        if self.state == RuntimeState.COOLDOWN:
            if now_s - self.last_trigger_time >= self.cooldown_seconds:
                self.state = RuntimeState.WAITING
            else:
                return None

        if conf < self.threshold:
            self.state = RuntimeState.WAITING
            return None

        if cls == self.last_class and (now_s - self.last_trigger_time) < self.cooldown_seconds:
            self.state = RuntimeState.COOLDOWN
            return None

        self.state = RuntimeState.DETECTING
        self.last_trigger_time = now_s
        self.last_class = cls
        self.state = RuntimeState.COOLDOWN
        return cls
