from __future__ import annotations

from typing import Any

from modeling.config import DatapackConfig, RunConfig
from modeling.data.text_train import TextPretrainDatapackConfig
from modeling.modules.action_module import ActionLIT, ActionLITConfig
from transformers.models.qwen2_5_omni import (
    Qwen2_5OmniProcessor,
)
from .qwen_25o_actions import Qwen2_5OmniThinkerForActionModelling


class Qwen25OActionLIT(ActionLIT):
    """
    Qwen-2.5O Lightning Module for text pretraining.
    Inherits from TextPretrainLIT and uses the Qwen-2.5O model.
    """

    def __init__(
        self,
        config: Qwen25OActionLITConfig,
        run_config: RunConfig,
    ):
        super().__init__(config=config, run_config=run_config)

        self.model = Qwen2_5OmniThinkerForActionModelling.from_pretrained(
            config.model_name,
            torch_dtype=self.dtype,
            attn_implementation=self.attn_impl,
        ).train()
        self.model_config = self.model.config


class Qwen25OActionLITConfig(ActionLITConfig):
    """
    Configuration class for Qwen-2.5O Lightning Module.
    Inherits from TextPretrainLITConfig and sets the model name.
    """

    config_path: str = "modeling.modules.action_instruct.qwen_25o.Qwen25OLITConfig"
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

    def create_module(self, run_config: RunConfig) -> Qwen25OActionLIT:
        return Qwen25OActionLIT(self, run_config)
