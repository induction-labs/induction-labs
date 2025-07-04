from __future__ import annotations

from pathlib import Path
from typing import Any

from modeling.config import DatapackConfig, RunConfig
from modeling.data.text_train import TextPretrainDatapackConfig
from modeling.modules.text_module import TextLIT, TextLITConfig
from transformers.models.qwen2_5_omni import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)


class Qwen25OLIT(TextLIT):
    """
    Qwen-2.5O Lightning Module for text pretraining.
    Inherits from TextPretrainLIT and uses the Qwen-2.5O model.
    """

    def __init__(
        self,
        config: Qwen25OLITConfig,
        run_config: RunConfig,
    ):
        super().__init__(config=config, run_config=run_config)

        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_impl,
        ).train()
        self.model_config = self.model.config


class Qwen25OLITConfig(TextLITConfig):
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
        tokenizer = processor.tokenizer
        return tokenizer

    def validate_datapack_compatibility(
        self, datapack_config: DatapackConfig[Any]
    ) -> TextPretrainDatapackConfig:
        assert isinstance(datapack_config, TextPretrainDatapackConfig), (
            f"Expected {datapack_config=} to be of type TextPretrainDatapackConfig"
        )
        return datapack_config

    def create_module(self, run_config: RunConfig, tmp_dir: Path) -> Qwen25OLIT:
        return Qwen25OLIT(self, run_config, tmp_dir)
