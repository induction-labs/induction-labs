from __future__ import annotations

from enum import Enum


class AttentionImplementation(str, Enum):
    """
    Enum for attention implementations.
    """

    FLASH_ATTENTION_2 = "flash_attention_2"
    SDPA = "sdpa"
    EAGER = "eager"
    MATH = "math"
