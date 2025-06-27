from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from env import Action


@dataclass
class TypingConfig:
    text: str
    wpm: int = 120  # words-per-minute


def typing_generator(cfg: TypingConfig, seed: int = 0):
    rng = np.random.default_rng(seed)
    ts = 0.0
    for ch in cfg.text:
        delay = 60 / (cfg.wpm * 5)  # rough char timing
        ts += delay * rng.uniform(0.8, 1.2)  # add a bit of jitter
        yield Action(ts=ts, key=ch)


@dataclass
class CursorPathConfig:
    path: list[tuple[int, int]]  # polyline in screen pixels
    duration: float  # seconds


def cursor_path_generator(cfg: CursorPathConfig, fps: int = 60):
    points = np.array(cfg.path)
    # linear interpolation to per-frame cursor positions
    total_frames = int(cfg.duration * fps)
    t = np.linspace(0, 1, total_frames)
    segments = np.linspace(0, 1, len(points))
    x = np.interp(t, segments, points[:, 0])
    y = np.interp(t, segments, points[:, 1])
    for i in range(total_frames):
        yield Action(ts=i / fps, cursor=(int(x[i]), int(y[i])))
