from __future__ import annotations

from importlib.util import find_spec

from modeling.types import AttentionImplementation


def get_attn_impl() -> AttentionImplementation:
    """
    Get the attention implementation based on available packages.

    Args:
        default (str): Default implementation if flash_attn is not available.
                      Common options: "sdpa", "eager", "math"

    Returns:
        str: "flash_attention_2" if flash_attn is available, otherwise the default
    """
    try:
        if find_spec("flash_attn") is None:
            return AttentionImplementation.SDPA
        return AttentionImplementation.FLASH_ATTENTION_2
    except ValueError:
        return (
            AttentionImplementation.SDPA
        )  # Default to SDPA if flash_attn is not available


def check_attn_impl(impl: AttentionImplementation) -> None:
    """
    Check if the specified attention implementation is available.

    Args:
        impl (AttentionImplementation): The attention implementation to check.

    Raises:
        ImportError: If the specified implementation is not available.
    """
    match impl:
        case AttentionImplementation.FLASH_ATTENTION_2:
            try:
                if find_spec("flash_attn") is None:
                    raise ImportError(
                        "Flash Attention 2 is not available. "
                        "Please install flash_attn package."
                    )
            except ImportError as e:
                raise ImportError(
                    "Flash Attention 2 is not available. "
                    "Please install flash_attn package."
                ) from e

        case _:
            # Don't validate other implementations
            pass
