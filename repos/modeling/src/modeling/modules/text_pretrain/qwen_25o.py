from __future__ import annotations

from modeling.config import RunConfig, InstanceConfig
from modeling.modules.text_pretrain.default import TextPretrainLITConfig
from transformers.models.qwen2_5_omni import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from modeling.modules.text_pretrain.default import (
    TextPretrainLIT,
)
from modeling.utils.class_property import class_property


class Qwen25OLIT(
    TextPretrainLIT[Qwen2_5OmniThinkerForConditionalGeneration, "Qwen25OLITConfig"]
):
    @class_property
    def model_cls(cls) -> type[Qwen2_5OmniThinkerForConditionalGeneration]:
        return Qwen2_5OmniThinkerForConditionalGeneration

    def init_model_meta(self, *args) -> Qwen2_5OmniThinkerForConditionalGeneration:
        return Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.module_config.model_name,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_impl,
        )


class Qwen25OLITConfig(TextPretrainLITConfig):
    """
    Configuration class for Qwen-2.5O Lightning Module.
    Inherits from TextPretrainLITConfig and sets the model name.
    """

    config_path: str = "modeling.modules.text_pretrain.qwen_25o.Qwen25OLITConfig"
    model_name: str = "Qwen/Qwen2.5-Omni-3B"
    tokenizer_name: str = "Qwen/Qwen2.5-Omni-3B"

    @property
    def get_tokenizer(self):
        processor = Qwen2_5OmniProcessor.from_pretrained(self.model_name)
        assert isinstance(processor, Qwen2_5OmniProcessor)
        tokenizer = processor.tokenizer  # type: ignore[attr-defined]
        return tokenizer

    def create_module(
        self,
        run_config: RunConfig,
        instance_config: InstanceConfig,
    ) -> Qwen25OLIT:
        return Qwen25OLIT(self, run_config, instance_config)
