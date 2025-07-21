from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from modeling.modules.base_module import (
    DATA_TYPE,
    MODEL_TYPE,
    BaseLITModule,
    BaseModuleConfig,
)

CONFIG_TYPE = TypeVar("CONFIG_TYPE", bound="TextLITConfig", covariant=True)


class TextLIT(BaseLITModule[MODEL_TYPE, DATA_TYPE, CONFIG_TYPE]):
    pass


class TextLITConfig(BaseModuleConfig):
    tokenizer_name: str

    @property
    @abstractmethod
    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        """
        Abstract property to get the tokenizer.
        Should be implemented in subclasses to return a tokenizer instance.
        """
        raise NotImplementedError("Subclasses must implement this method.")
