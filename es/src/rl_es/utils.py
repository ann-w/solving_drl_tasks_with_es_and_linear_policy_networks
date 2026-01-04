import time
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import gymnasium as gym
from skimage.measure import block_reduce


@dataclass
class ExperimentTimer:
    """Track and save experiment timing information."""
    
    data_folder: Optional[str] = None
    start_time: float = field(default_factory=time.perf_counter)
    start_wall_time: float = field(default_factory=time.time)
    checkpoints: dict = field(default_factory=dict)
    
    def checkpoint(self, name: str) -> float:
        """Record a named checkpoint and return elapsed time since start."""
        elapsed = time.perf_counter() - self.start_time
        self.checkpoints[name] = elapsed
        return elapsed
    
    def elapsed(self) -> float:
        """Return elapsed time in seconds since start."""
        return time.perf_counter() - self.start_time
    
    def elapsed_formatted(self) -> str:
        """Return elapsed time as human-readable string."""
        elapsed = self.elapsed()
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def summary(self) -> dict:
        """Return timing summary dictionary."""
        total_elapsed = self.elapsed()
        return {
            "total_seconds": total_elapsed,
            "total_formatted": self.elapsed_formatted(),
            "start_timestamp": self.start_wall_time,
            "end_timestamp": time.time(),
            "checkpoints": self.checkpoints,
        }
    
    def save(self, filename: str = "timing.json") -> None:
        """Save timing information to JSON file."""
        if self.data_folder is not None:
            filepath = os.path.join(self.data_folder, filename)
            with open(filepath, "w") as f:
                json.dump(self.summary(), f, indent=2)
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.start_wall_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.checkpoint("end")
        self.save()


def softmax(x):
    x_exp = np.exp(x - np.max(x))
    return x_exp / x_exp.sum()


def argmax(x):
    return np.argmax(x, axis=1)


def identity(x):
    return x


def clip(lb, ub):
    def inner(x):
        return np.clip(x, lb, ub)

    return inner


def uint8tofloat(obs):
    return ((obs.astype(float) / 255) * 2) - 1
