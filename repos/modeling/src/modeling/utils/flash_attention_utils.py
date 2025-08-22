import inspect

from synapse.utils.logging import configure_logging, logging

from modeling.types.attn_impl import AttentionImplementation

# from transformers.masking_utils import AttentionMaskInterface, flash_attention_mask
# from transformers.integrations.flash_attention import flash_attention_forward
# from transformers.modeling_utils import AttentionInterface

logger = configure_logging(__name__, level=logging.DEBUG)


def _custom_lazy_imports(impl: AttentionImplementation):
    # returns funcs and pad/unpad based on impl
    if impl == AttentionImplementation.FLASH_ATTENTION_2:
        logger.debug("Using default flash attention 2 implementation")
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import pad_input, unpad_input

        return (
            flash_attn_func,
            flash_attn_varlen_func,
            pad_input,
            unpad_input,
            False,
        )
    if impl == AttentionImplementation.FLASH_ATTENTION_2_CUTE:
        logger.debug(
            "Custom importing flash attention 2. Using 🥰 CUTE 🥰 Flash Attention Kernel"
        )
        from flash_attn.bert_padding import pad_input, unpad_input
        from flash_attn.cute.interface import (
            flash_attn_func,
        )

        from modeling.utils.flash_attention import hybrid_varlen_attention

        return (
            flash_attn_func,
            hybrid_varlen_attention,
            pad_input,
            unpad_input,
            False,
        )
    raise NotImplementedError(
        f"Unsupported flash attention implementation: {impl}. "
        "Only FLASH_ATTENTION_2 and FLASH_ATTENTION_2_CUTE are supported."
    )

    # if impl == "flash_attention_3":
    #     from flash_attn_interface import (  # type: ignore[import]
    #         flash_attn_func,
    #         flash_attn_varlen_func,
    #     )

    #     pad_input, unpad_input = _fa3_pad_input, _fa3_unpad_input
    #     return flash_attn_func, flash_attn_varlen_func, pad_input, unpad_input, True
    # else:
    #     pad_input, unpad_input = _fa3_pad_input, _fa3_unpad_input
    #     return (
    #         getattr(impl, "flash_attn_func", None),
    #         impl.flash_attn_varlen_func,  # type: ignore[union-attr]
    #         pad_input,
    #         unpad_input,
    #         True,
    #     )


def configure_flash_attention(impl: AttentionImplementation):
    import transformers.modeling_flash_attention_utils as fa_utils

    assert impl in (
        AttentionImplementation.FLASH_ATTENTION_2,
        AttentionImplementation.FLASH_ATTENTION_2_CUTE,
    ), (
        f"Unsupported flash attention implementation: {impl}. Only FLASH_ATTENTION_2 is supported."
    )
    if any(
        k in fa_utils.__dict__
        for k in ("_flash_fn", "_flash_varlen_fn", "_pad_fn", "_unpad_fn", "_is_fa3")
    ):
        logger.warning(
            "Flash attention is already configured. Reconfiguring may lead to unexpected behavior."
        )
    flash_fn, flash_varlen_fn, pad_fn, unpad_fn, is_fa3 = _custom_lazy_imports(impl)
    fa_utils.__dict__["_flash_fn"] = flash_fn
    fa_utils.__dict__["_flash_varlen_fn"] = flash_varlen_fn
    fa_utils.__dict__["_pad_fn"] = pad_fn
    fa_utils.__dict__["_unpad_fn"] = unpad_fn
    fa_utils.__dict__["_is_fa3"] = is_fa3
    flash_supports_window = (
        "window_size" in inspect.signature(flash_varlen_fn).parameters
    )
    fa_utils.__dict__["_flash_supports_window"] = flash_supports_window

    # We don't need to register the attention here because fa2cute is aliased as flash_attention_2
    # AttentionInterface.register(impl, flash_attention_forward)
    # AttentionMaskInterface.register(impl, flash_attention_mask)
