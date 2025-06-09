from __future__ import annotations

from enum import Enum

import torch


class DType(str, Enum):
    """
    Enum for data types used in modeling.
    """

    bf16 = "bfloat16"
    fp16 = "float16"
    fp32 = "float32"

    @property
    def torch_dtype(self) -> torch.dtype:
        """
        Convert the enum value to a PyTorch dtype.
        """
        match self:
            case DType.bf16:
                return torch.bfloat16
            case DType.fp16:
                return torch.float16
            case DType.fp32:
                return torch.float32
            case _:
                raise ValueError(f"Unsupported dtype: {self}")
