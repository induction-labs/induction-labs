from __future__ import annotations

from abc import ABC, abstractmethod

import lightning as L
import torch
from modeling.config import ModuleConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class TextLIT(ABC, L.LightningModule):
    model: PreTrainedModel
    model_config: PretrainedConfig

    # @abstractmethod
    def __init__(self, config: TextLITConfig):
        super().__init__()
        self.config = config

    def training_step(self, inputs):
        # Forward pass through the model
        outputs = self.model.forward(**inputs)
        assert isinstance(outputs.loss, torch.Tensor), (
            f"Expected outputs.loss to be a Tensor, got {type(outputs.loss)}"
        )
        # TODO: Add more metrics and logging (steptime, tok/s, etc.)

        self.log(
            "train/loss",
            outputs.loss,
            logger=True,
            # rank_zero_only=True
        )
        return outputs.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr, fused=True
        )
        return optimizer


class TextLITConfig(ModuleConfig):
    model_name: str
    tokenizer_name: str
    lr: float

    @property
    @abstractmethod
    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        """
        Abstract property to get the tokenizer.
        Should be implemented in subclasses to return a tokenizer instance.
        """
        raise NotImplementedError("Subclasses must implement this method.")
