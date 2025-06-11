from __future__ import annotations

from abc import ABC, abstractmethod

import lightning as L
import torch
from modeling.config import ModuleConfig
from modeling.types.attn_impl import AttentionImplementation
from modeling.types.dtype import DType
from modeling.utils.elapsed_timer import elapsed_timer
from modeling.utils.get_attn_impl import check_attn_impl
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class TextLIT(ABC, L.LightningModule):
    model: PreTrainedModel
    model_config: PretrainedConfig

    # @abstractmethod
    def __init__(self, config: TextLITConfig):
        super().__init__()
        self.attn_impl = config.attn_impl
        self._dtype = config.dtype.torch_dtype
        check_attn_impl(self.attn_impl)
        print(f"Using attention implementation: {self.attn_impl}, {self.dtype=}")
        self.config = config
        torch.cuda.reset_peak_memory_stats()

    def training_step(self, inputs):
        # Forward pass through the model

        with elapsed_timer() as timer:
            # Note: You CANT call self.model.forward here because it fucking doesn't trigger the FSDP hooks so weights dont gather
            outputs = self.model(**inputs)
            elapsed = timer()

        assert isinstance(outputs.loss, torch.Tensor), (
            f"Expected outputs.loss to be a Tensor, got {type(outputs.loss)}"
        )
        # TODO: Add more metrics and logging (steptime, tok/s, etc.)
        torch.cuda.synchronize()
        allocated_memory = torch.cuda.memory_allocated(
            device=torch.cuda.current_device()
        )
        reserved_memory = torch.cuda.memory_reserved(device=torch.cuda.current_device())

        metrics = {
            "train/step_time": elapsed,
            "train/tokens_per_second": inputs["input_ids"].numel() / elapsed,
            "train/loss": outputs.loss,
            "train/allocated_memory": allocated_memory / 1e9,  # in GB
            "train/reserved_memory": reserved_memory / 1e9,  # in GB
        }

        self.log_dict(
            metrics,
            logger=True,
        )
        return outputs.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            # foreach=True,
            fused=True,
        )
        return optimizer


class TextLITConfig(ModuleConfig):
    model_name: str
    tokenizer_name: str
    lr: float
    attn_impl: AttentionImplementation = AttentionImplementation.SDPA
    dtype: DType = DType.bf16

    @property
    @abstractmethod
    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        """
        Abstract property to get the tokenizer.
        Should be implemented in subclasses to return a tokenizer instance.
        """
        raise NotImplementedError("Subclasses must implement this method.")
