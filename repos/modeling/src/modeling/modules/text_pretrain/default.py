from __future__ import annotations

from typing import Any

from modeling.config import DatapackConfig
from modeling.data.text_train import TextPretrainDatapackConfig
from modeling.modules.text_module import TextLIT, TextLITConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.modeling_utils import PreTrainedModel


class TextPretrainLIT(TextLIT):
    def __init__(self, config: TextPretrainLITConfig):
        super().__init__(config)

        self.model_config = AutoConfig.from_pretrained(
            self.config.model_name, use_cache=False
        )
        model = AutoModelForCausalLM.from_config(
            self.model_config, torch_dtype=self._dtype
        )
        assert isinstance(model, PreTrainedModel)
        self.model = model.train()


class TextPretrainLITConfig(TextLITConfig):
    config_path: str = "modeling.modules.text_pretrain.default.TextPretrainLITConfig"
    model_name: str = "openai-community/gpt2"
    tokenizer_name: str = "openai-community/gpt2"
    lr: float = 1e-3

    @property
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.tokenizer_name)

    def validate_datapack_compatibility(
        self, datapack_config: DatapackConfig[Any]
    ) -> TextPretrainDatapackConfig:
        assert isinstance(datapack_config, TextPretrainDatapackConfig), (
            f"Expected {datapack_config=} to be of type TextPretrainDatapackConfig"
        )
        return datapack_config

    def create_module(self) -> TextPretrainLIT:
        return TextPretrainLIT(self)
