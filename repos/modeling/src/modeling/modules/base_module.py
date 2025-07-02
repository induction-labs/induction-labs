from __future__ import annotations

from abc import ABC, abstractmethod

import lightning as L
import torch
from modeling.config import ModuleConfig, RunConfig
from modeling.utils.elapsed_timer import elapsed_timer
from modeling.utils.get_attn_impl import check_attn_impl
from torch.distributed.device_mesh import DeviceMesh
from transformers.configuration_utils import PretrainedConfig

from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from transformers.modeling_utils import PreTrainedModel
from typing import Generic, TypeVar, final

MODEL_TYPE = TypeVar("MODEL_TYPE", bound=PreTrainedModel, covariant=True)
DATA_TYPE = TypeVar("DATA_TYPE")
CONFIG_TYPE = TypeVar("CONFIG_TYPE", bound=ModuleConfig, covariant=True)


class BaseLITModule(
    ABC, L.LightningModule, Generic[MODEL_TYPE, DATA_TYPE, CONFIG_TYPE]
):
    model: MODEL_TYPE

    @property
    def model_config(self) -> PretrainedConfig:
        return self.model.config

    def __init__(self, module_config: CONFIG_TYPE, run_config: RunConfig):
        super().__init__()
        self.module_config = module_config
        self.run_config = run_config
        self.attn_impl = run_config.attn_impl
        self._dtype = run_config.precision.torch_dtype

        check_attn_impl(self.attn_impl)
        print(f"Using attention implementation: {self.attn_impl}, {self.dtype=}")

        if self.run_config.accelerator == "cuda":
            torch.cuda.reset_peak_memory_stats()

    @abstractmethod
    def shard_model(
        self,
        *,
        mp_policy: MixedPrecisionPolicy,
        device_mesh: DeviceMesh,
    ) -> FSDPModule:
        """
        Abstract method to be implemented by subclasses for sharding the model.
        This method should handle the Fully Sharded Data Parallel (FSDP) setup.
        Returns FSDPModule
        """
        dp_mesh = device_mesh["data_parallel"]  # provided by ModelParallelStrategy
        fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
        return fully_shard(self.model, **fsdp_config)

    @final
    def configure_model(self) -> None:
        # We need to ensure that all models are fsdp because that is we use by default

        if isinstance(self.model, FSDPModule):
            return  # already configured

        assert isinstance(self.device_mesh, DeviceMesh), (
            f"Expected device_mesh to be a DeviceMesh, got {type(self.device_mesh)}"
        )
        self.model = self.shard_model(
            mp_policy=self.run_config.mp_policy,
            device_mesh=self.device_mesh,
        )  # type: ignore[assignment]
        assert isinstance(self.model, FSDPModule), (
            f"Expected self.model to be a FullyShardedDataParallel, got {type(self.model)}"
        )

    @abstractmethod
    def run_training_step(self, inputs: DATA_TYPE) -> torch.Tensor:
        """
        Abstract method to be implemented by subclasses for the training step.
        This method should handle the forward pass and return the loss.
        Note: You CANT call self.model.forward here because it fucking doesn't
          trigger the FSDP hooks so weights dont gather
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def check_loss(loss: torch.Tensor) -> torch.Tensor:
        assert loss.ndim == 0, f"Expected loss to be a scalar tensor, got {loss.ndim}"

        # Assert loss is not NaN or Inf
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"
        return loss

    @final
    def training_step(self, inputs: DATA_TYPE):
        # Forward pass through the model

        with elapsed_timer() as timer:
            loss = self.run_training_step(inputs)
            loss = self.check_loss(loss)
            elapsed = timer()

        # TODO: Add more metrics and logging (steptime, tok/s, etc.)
        memory_metrics = {}
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
            # "train/tokens_per_second": inputs["input_ids"].numel() / elapsed,
            "train/loss": loss,
            **memory_metrics,
        }

        self.log_dict(
            metrics,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.run_config.lr,
            # foreach=True,
            fused=True,
        )
        return optimizer
