from __future__ import annotations

from enum import Enum


class AttentionImplementation(str, Enum):
    """
    Enum for attention implementations.
    """

    FLASH_ATTENTION_2 = "flash_attention_2"
    FLASH_ATTENTION_2_CUTE = "flash_attention_2_cute"

    @property
    def is_flash_attention(self) -> bool:
        return self in (
            AttentionImplementation.FLASH_ATTENTION_2,
            AttentionImplementation.FLASH_ATTENTION_2_CUTE,
        )

    @property
    def hf_string(self) -> str:
        # We don't use the attention actual name because we need to alias flash attention2 cute
        # to flash_attention_2 because alot of HF things are hardcoded to use that
        if self == AttentionImplementation.FLASH_ATTENTION_2_CUTE:
            return "flash_attention_2"
        return self

    SDPA = "sdpa"
    EAGER = "eager"
    MATH = "math"
