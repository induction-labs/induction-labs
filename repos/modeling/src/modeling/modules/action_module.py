from __future__ import annotations

from abc import ABC

import lightning as L
import torch
from modeling.config import ModuleConfig, RunConfig
from modeling.utils.elapsed_timer import elapsed_timer
from modeling.utils.get_attn_impl import check_attn_impl
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


class ActionLIT(ABC, L.LightningModule):
    model: PreTrainedModel
    model_config: PretrainedConfig

    def __init__(self, config: ActionLITConfig, run_config: RunConfig):
        super().__init__()
        self.attn_impl = run_config.attn_impl
        self._dtype = run_config.precision.torch_dtype
        check_attn_impl(self.attn_impl)
        print(f"Using attention implementation: {self.attn_impl}, {self.dtype=}")
        self.config = config
        self.run_config = run_config

        if self.run_config.accelerator == "cuda":
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

        if self.run_config.accelerator == "cuda":
            torch.cuda.synchronize()

            allocated_memory = torch.cuda.memory_allocated(
                device=torch.cuda.current_device()
            )
            reserved_memory = torch.cuda.memory_reserved(
                device=torch.cuda.current_device()
            )
            memory_metrics = {
                "train/allocated_memory": allocated_memory / 1e9,  # in GB
                "train/reserved_memory": reserved_memory / 1e9,  # in GB
            }

        metrics = {
            "train/step_time": elapsed,
            "train/tokens_per_second": inputs["input_ids"].numel() / elapsed,
            "train/loss": outputs.loss,
            **memory_metrics,
        }

        self.log_dict(
            metrics,
            logger=True,
        )
        return outputs.loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.run_config.lr,
            # foreach=True,
            fused=True,
        )
        return optimizer


class ActionLITConfig(ModuleConfig):
    model_name: str
