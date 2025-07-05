from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from accelerate import load_checkpoint_and_dispatch
import lightning as L
from modeling.utils.cloud_path import CloudPath
import torch
from modeling.config import ModuleConfig, RunConfig
from modeling.utils.elapsed_timer import elapsed_timer
from modeling.utils.get_attn_impl import check_attn_impl
from torch.distributed.device_mesh import DeviceMesh
from transformers.configuration_utils import PretrainedConfig

from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from transformers.modeling_utils import PreTrainedModel
from typing import Generic, TypeVar, cast, final
from synapse.utils.logging import configure_logging
import os
from modeling.checkpoints.load import download_model_checkpoint

logger = configure_logging(
    __file__,
    #    level=logging.DEBUG
)


class BaseModuleConfig(ModuleConfig):
    """
    Base configuration class for modules.
    This class should be extended by specific module configurations.
    """

    model_name: str
    checkpoint_path: CloudPath | None = None

    @abstractmethod
    def create_module(
        self, run_config: RunConfig, tmp_dir: Path
    ) -> "BaseLITModule": ...


MODEL_TYPE = TypeVar("MODEL_TYPE", bound=PreTrainedModel, covariant=True)
DATA_TYPE = TypeVar("DATA_TYPE")
CONFIG_TYPE = TypeVar("CONFIG_TYPE", bound=BaseModuleConfig, covariant=True)


class BaseLITModule(
    ABC, L.LightningModule, Generic[MODEL_TYPE, DATA_TYPE, CONFIG_TYPE]
):
    model: MODEL_TYPE

    @property
    def model_config(self) -> PretrainedConfig:
        return self.model.config

    @final
    def __init__(
        self, module_config: CONFIG_TYPE, run_config: RunConfig, tmp_dir: Path
    ):
        super().__init__()
        self.module_config = module_config
        self.run_config = run_config
        self.attn_impl = run_config.attn_impl
        self._dtype = run_config.precision.torch_dtype
        self.tmp_dir = tmp_dir
        os.makedirs(self.tmp_dir, exist_ok=True)

        check_attn_impl(self.attn_impl)
        logger.debug(f"Using attention implementation: {self.attn_impl}, {self.dtype=}")
        if self.run_config.accelerator == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.model = self.init_model_meta()
        self.download_weights(self.tmp_dir)

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
        logger.debug("Configuring model for FSDP sharding...")

        if isinstance(self.model, FSDPModule):
            return  # already configured
        self.model = self.load_weights(self.tmp_dir)

        assert isinstance(self.device_mesh, DeviceMesh), (
            f"Expected device_mesh to be a DeviceMesh, got {type(self.device_mesh)}"
        )
        self.model = self.shard_model(
            mp_policy=self.run_config.mp_policy,
            device_mesh=self.device_mesh,
        )  # type: ignore[assignment]
        logger.debug(f"Sharded model {self.model} with dtype {self.dtype}")
        assert isinstance(self.model, FSDPModule), (
            f"Expected self.model to be a FullyShardedDataParallel, got {type(self.model)}"
        )

    @abstractmethod
    def init_model_meta(self) -> MODEL_TYPE:
        """
        Abstract method to be implemented by subclasses for initializing the model in meta mode.
        This method should return a model instance on meta device
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def download_weights(self, tmpdir: Path) -> None:
        """
        Abstract method to be implemented by subclasses for downloading model weights.
        """
        download_model_checkpoint(
            tmpdir,
            self.module_config.model_name,
            self.module_config.checkpoint_path,
        )

    def load_weights(self, tmpdir: Path) -> MODEL_TYPE:
        """
        Load the model weights from the specified checkpoint directory.
        This method should handle the loading of pre-trained weights or checkpoint files.
        """
        logger.debug(f"Loading model weights from {tmpdir}")
        # Load the model weights and dispatch them to the appropriate devices
        self.model.to_empty(device=torch.cuda.current_device())
        loaded_model = load_checkpoint_and_dispatch(
            self.model,
            checkpoint=tmpdir,  # hub ID or local folder
            device_map={"": torch.cuda.current_device()},
            dtype=self.model.dtype,
        )
        return cast(MODEL_TYPE, loaded_model)

    @abstractmethod
    def run_validation_step(
        self, inputs: DATA_TYPE
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Abstract method to be implemented by subclasses for the evaluation step.
        This method should handle the forward pass and return the loss and metrics.
        Note: You CANT call self.model.forward here because it fucking doesn't
          trigger the FSDP hooks so weights dont gather
        """
        raise NotImplementedError("Subclasses must implement this method.")

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

    @final
    def validation_step(self, inputs: DATA_TYPE):
        # Forward pass through the model
        with elapsed_timer() as timer:
            loss, val_metrics = self.run_validation_step(inputs)
            loss = self.check_loss(loss)
            elapsed = timer()

        metrics = {
            "val/step_time": elapsed,
            "val/loss": loss,
            **val_metrics,
        }

        return metrics

    def validation_epoch_end(self, validation_step_outputs):
        metrics = {}
        for output in validation_step_outputs:
            for key, value in output.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
        # Average the metrics across all validation steps
        for key, value in metrics.items():
            metrics[key] = torch.tensor(value).mean()

        self.log_dict(
            metrics,
            logger=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.run_config.lr,
            # foreach=True,
            fused=True,
        )
        return optimizer
