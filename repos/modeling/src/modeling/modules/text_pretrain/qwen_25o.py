from __future__ import annotations

from modeling.modules.text_module import TextLIT
from transformers.models.qwen2_5_omni import (
    Qwen2_5OmniConfig,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

from .default import TextPretrainLITConfig


class Qwen25OLIT(TextLIT):
    """
    Qwen-2.5O Lightning Module for text pretraining.
    Inherits from TextPretrainLIT and uses the Qwen-2.5O model.
    """

    def __init__(
        self,
        config: TextPretrainLITConfig,
    ):
        super().__init__(config=config)
        self.model_config = Qwen2_5OmniConfig.from_pretrained(config.model_name)
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            config.model_name,
        ).train()


class Qwen25OLITConfig(TextPretrainLITConfig):
    """
    Configuration class for Qwen-2.5O Lightning Module.
    Inherits from TextPretrainLITConfig and sets the model name.
    """

    config_path: str = "modeling.modules.text_pretrain.qwen_25o.Qwen25OLITConfig"
    model_name: str = "Qwen/Qwen-2.5-0"
    tokenizer_name: str = "Qwen/Qwen-2.5-0"

    def create_module(self) -> Qwen25OLIT:
        return Qwen25OLIT(self)
