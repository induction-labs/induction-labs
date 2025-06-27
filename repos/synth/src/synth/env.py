from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CURSOR_RADIUS = 6


@dataclass
class Action:
    ts: float  # seconds
    cursor: tuple[int, int] | None = None
    key: str | None = None  # single char


class ScreenshotEnv:
    """Draws actions on top of a background screenshot."""

    def __init__(self, bg_path: Path, font_path: Path | None = None):
        self.bg = Image.open(bg_path).convert("RGB")
        self.font = ImageFont.truetype(font_path, 20) if font_path else None
        self.reset()

    def reset(self):
        self.frame = self.bg.copy()
        self.cursor_pos = None
        self.text = ""

    def step(self, action: Action) -> np.ndarray:
        # fresh copy each frame
        self.frame = self.bg.copy()
        draw = ImageDraw.Draw(self.frame)

        if action.cursor:
            self.cursor_pos = action.cursor
        if action.key:
            self.text += action.key

        # draw typed text
        if self.text and self.font:
            draw.text(
                (20, self.bg.height - 40), self.text, font=self.font, fill="black"
            )

        # draw cursor as a small circle
        if self.cursor_pos:
            x, y = self.cursor_pos
            draw.ellipse(
                (
                    x - CURSOR_RADIUS,
                    y - CURSOR_RADIUS,
                    x + CURSOR_RADIUS,
                    y + CURSOR_RADIUS,
                ),
                fill="red",
            )
        return np.asarray(self.frame, dtype=np.uint8)
