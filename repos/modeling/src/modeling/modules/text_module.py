from __future__ import annotations

from abc import abstractmethod

from modeling.config import ModuleConfig, _LitModule


class TextLITConfig(ModuleConfig[_LitModule]):
    @property
    @abstractmethod
    def get_tokenizer(self):
        """
        Abstract property to get the tokenizer.
        Should be implemented in subclasses to return a tokenizer instance.
        """
        raise NotImplementedError("Subclasses must implement this method.")
