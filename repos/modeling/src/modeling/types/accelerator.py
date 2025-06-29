from __future__ import annotations

from enum import Enum


class Accelerator(str, Enum):
    """
    Enum for accelerator types used in modeling.
    """

    CPU = "cpu"
    CUDA = "cuda"
