from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
from pydantic import BaseModel, Field


@dataclass
class Cubic:
    # specify a cubic that passes through (0, 0) and (1, a)
    # with tangent at (0, 0) being m and tangent at (1, m) being n
    # can convert to a polynomial with
    # f(x) = (m+n-2a)x^3 + (3a-m-2n)x^2 + nx
    m: float
    n: float
    a: float

    def coeffs(self) -> tuple[float, float, float, float]:
        """Return the four polynomial coefficients (c3, c2, c1, c0)."""
        c3 = self.m + self.n - 2 * self.a
        c2 = 3 * self.a - 2 * self.m - self.n
        c1 = self.m
        c0 = 0.0
        return c3, c2, c1, c0

    def to_ndarray(self) -> np.ndarray:
        """Return the coefficients as a numpy array."""
        return np.array([self.m, self.n, self.a])

    def __call__(self, x: float):
        """Evaluate the cubic at x (scalar or array)."""
        c3, c2, c1, _ = self.coeffs()
        return ((c3 * x + c2) * x + c1) * x  # Horner form


# Mouse action models
class MouseMove(BaseModel):
    action: Literal["mouse_move"] = "mouse_move"
    x: Union[int, float]  # noqa: UP007
    y: Union[int, float]  # noqa: UP007


class MouseButton(BaseModel):
    action: Literal["mouse_button"] = "mouse_button"
    button: str #Literal["left", "right", "middle"]
    x: int
    y: int
    is_down: bool


class Scroll(BaseModel):
    action: Literal["scroll"] = "scroll"
    delta_x: int
    delta_y: int

    x: int
    y: int


# Keyboard action models
class KeyButton(BaseModel):
    action: Literal["key_button"] = "key_button"
    key: str
    is_down: bool


actionType = Union[MouseMove, MouseButton, Scroll, KeyButton]  # noqa: UP007


class Action(BaseModel):
    action: actionType = Field(discriminator="action")
    timestamp: float

    @staticmethod
    def from_action_type(action: actionType, timestamp: float | None = None) -> Action:
        return Action(
            action=action,
            timestamp=time.time_ns() * 1e-9 if timestamp is None else timestamp,
        )
