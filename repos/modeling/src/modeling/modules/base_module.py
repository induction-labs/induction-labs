from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from functools import partial

from modeling.utils.cloud_path import CloudPath
import torch
from modeling.config import (
    DistributedInstanceConfig,
    ModuleConfig,
    RunConfig,
    GlobalState,
    RuntimeConfig,
)
from modeling.utils.elapsed_timer import elapsed_timer
from modeling.utils.get_attn_impl import check_attn_impl
from torch.distributed.device_mesh import DeviceMesh
from transformers.configuration_utils import PretrainedConfig

from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from transformers.modeling_utils import PreTrainedModel
from typing import Any, Generic, Literal, TypeVar, cast, final
from synapse.utils.logging import configure_logging
import os
from modeling.checkpoints.load import download_model_checkpoint
from pydantic import BaseModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import logging
from modeling.utils.class_property import class_property


logger = configure_logging(
    __file__,
    level=logging.DEBUG,  # Set to DEBUG for more verbose output
)


def lr_lambda(
    step: int, warmup_steps: int, end_steps: int, start_lr: float, end_lr: float
):
    if step < warmup_steps:
        return step / warmup_steps  # scales 0 → 1
    else:
        progress = (step - warmup_steps) / max(1, end_steps - warmup_steps)
        progress = min(progress, 1.0)  # clamp to 1.0
        return 1 + (end_lr / start_lr - 1) * progress  # 1 → end_lr/start_lr


class CompileConfig(BaseModel):
    class CompileMode(str, Enum):
        DEFAULT = "default"
        REDUCE_OVERHEAD = "reduce-overhead"
        MAX_AUTOTUNE = "max-autotune"

    mode: CompileMode = CompileMode.DEFAULT
    fullgraph: bool = False


class ActivationCheckpointConfig(BaseModel):
    """
    Configuration for activation checkpointing.
    This class is used to configure the activation checkpointing settings for an experiment.
    """

    # Need to include a field so that it serializes
    layers: Literal["all"] = "all"


class BaseModuleConfig(ModuleConfig):
    """
    Base configuration class for modules.
    This class should be extended by specific module configurations.
    """

    model_name: str
    checkpoint_path: CloudPath | None = None
    compile: CompileConfig | None = None
    activation_checkpointing: ActivationCheckpointConfig | None = (
        ActivationCheckpointConfig()
    )

    @abstractmethod
    def create_module(
        self,
        run_config: RunConfig,
        runtime_config: RuntimeConfig,
        instance_config: DistributedInstanceConfig,
    ) -> "BaseLITModule": ...


MODEL_TYPE = TypeVar("MODEL_TYPE", bound=PreTrainedModel, covariant=True)
DATA_TYPE = TypeVar("DATA_TYPE")
CONFIG_TYPE = TypeVar("CONFIG_TYPE", bound=BaseModuleConfig, covariant=True)


def get_mem_stats(device=None):
    mem = torch.cuda.memory_stats(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "total_gb": 1e-9 * props.total_memory,
        "curr_alloc_gb": 1e-9 * mem["allocated_bytes.all.current"],
        "peak_alloc_gb": 1e-9 * mem["allocated_bytes.all.peak"],
        "curr_resv_gb": 1e-9 * mem["reserved_bytes.all.current"],
        "peak_resv_gb": 1e-9 * mem["reserved_bytes.all.peak"],
    }


class LRSchedulerConfig(BaseModel):
    """Configuration for learning rate scheduler."""

    model_config = {"arbitrary_types_allowed": True}

    scheduler: LRScheduler
    interval: str = "step"  # "step" or "epoch"


class OptimizerConfig(BaseModel):
    """Configuration for optimizer and learning rate scheduler."""

    model_config = {"arbitrary_types_allowed": True}

    optimizer: Optimizer
    lr_scheduler: LRSchedulerConfig


class BaseLITModule(ABC, Generic[MODEL_TYPE, DATA_TYPE, CONFIG_TYPE]):
    model: MODEL_TYPE

    @class_property
    @abstractmethod
    def model_cls(cls) -> type[MODEL_TYPE]:
        """
        Class property that should return the model class type.
        This is used to instantiate the model in meta mode and load weights.
        """
        pass

    @property
    def device(self) -> torch.device:
        """
        Returns the current device of the model.
        This is useful for ensuring that the model and data are on the same device.
        """
        return self.instance_config.device

    @property
    def model_config(self) -> PretrainedConfig:
        return self.model.config

    @property
    def tmp_dir(self) -> Path:
        """
        Returns the temporary directory path where model weights and checkpoints are stored.
        This is useful for downloading and loading model weights.
        """
        return self.runtime_config.tmp_dir / "module"

    @final
    def __init__(
        self,
        module_config: CONFIG_TYPE,
        run_config: RunConfig,
        runtime_config: RuntimeConfig,
        instance_config: DistributedInstanceConfig,
    ):
        self.runtime_config = runtime_config
        self.instance_config = instance_config
        self.module_config = module_config
        self.run_config = run_config
        self.attn_impl = run_config.attn_impl
        self.dtype = run_config.precision.torch_dtype
        os.makedirs(self.tmp_dir, exist_ok=True)

        check_attn_impl(self.attn_impl)
        logger.debug(f"Using attention implementation: {self.attn_impl}, {self.dtype=}")
        if self.run_config.accelerator == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.model = self.init_model_meta()
        # TODO: Move this to a separate method
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
        # dp_mesh = device_mesh["data_parallel"]  # provided by ModelParallelStrategy
        # fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
        # return fully_shard(self.model, **fsdp_config)
        return fully_shard(self.model)

    @final
    def configure_model(self) -> None:
        # We need to ensure that all models are fsdp because that is we use by default
        logger.debug("Configuring model for FSDP sharding...")

        if isinstance(self.model, FSDPModule):
            return  # already configured
        self.model = self.load_weights(self.tmp_dir)
        if self.module_config.activation_checkpointing is not None:
            logger.debug("Enabling activation checkpointing...")
            # Enable activation checkpointing if configured
            self.model.gradient_checkpointing_enable()  # Hypothetical method, replace with actual implementation

        # for layer_id, transformer_block in enumerate(self.model.model.layers):
        #     # Apply activation checkpointing

        #     # For now this is broken with HF models https://github.com/huggingface/transformers/issues/34928

        #     transformer_block = checkpoint_wrapper(transformer_block)
        #     self.model.model.layers[layer_id] = transformer_block

        if self.module_config.compile is not None:
            self.model = torch.compile(
                self.model,
                mode=self.module_config.compile.mode,
                fullgraph=self.module_config.compile.fullgraph,
            )  # type: ignore[assignment]

        # assert isinstance(self.device_mesh, DeviceMesh), (
        #     f"Expected device_mesh to be a DeviceMesh, got {type(self.device_mesh)}"
        # )
        # self.model = self.shard_model(
        #     mp_policy=self.run_config.mp_policy,
        #     device_mesh=self.device_mesh,
        # )  # type: ignore[assignment]
        # logger.debug(f"Sharded model {self.model} with dtype {self.dtype}")
        # assert isinstance(self.model, FSDPModule), (
        #     f"Expected self.model to be a FullyShardedDataParallel, got {type(self.model)}"
        # )

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
        logger.debug(f"Loading model weights from {tmpdir} to device {self.device}")
        # Load the model weights and dispatch them to the appropriate devices
        self.model.to_empty(device=self.device)
        # For now we load from hf only bc load_checkpoint_and_dispatch is broken
        # safetensors_rust.SafetensorError: device cuda is invalid

        loaded_model = self.model_cls.from_pretrained(
            self.module_config.model_name,
            config=self.model.config,
            torch_dtype=self.dtype,
            device_map={
                "": self.device  # Use the device index for the model
            },  # Ensure model is loaded on the correct device
            attn_implementation=self.attn_impl,
        )

        # loaded_model = load_checkpoint_and_dispatch(
        #     self.model,
        #     checkpoint=tmpdir,  # hub ID or local folder
        #     device_map={"": self.device},
        #     dtype=self.model.dtype,
        # )

        return cast(MODEL_TYPE, loaded_model)

    @final
    def wandb_log(self, global_state: GlobalState, metrics: dict[str, Any]) -> None:
        """
        Log metrics to Wandb if available.
        This method is a wrapper around the logger's log_dict method.
        """
        # logger.debug(metrics)
        if global_state.wandb is not None:
            global_state.wandb.log(metrics)

    @abstractmethod
    def run_validation_step(
        self,
        inputs: DATA_TYPE,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
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
    def training_step(
        self,
        inputs: DATA_TYPE,
        global_state: GlobalState,
    ):
        # Forward pass through the model
        assert self.model.training, (
            f"Expected model to be in training mode, got {self.model.training}"
        )
        with elapsed_timer() as timer:
            loss = self.run_training_step(inputs)
            loss = self.check_loss(loss)
            elapsed = timer()

        # TODO: Add more metrics and logging (steptime, tok/s, etc.)
        memory_metrics = {}
        if self.run_config.accelerator == "cuda":
            torch.cuda.synchronize()
            memory_metrics = get_mem_stats(device=torch.cuda.current_device())
            torch.cuda.reset_peak_memory_stats()

        metrics = {
            "train/step_time": elapsed,
            # "train/tokens_per_second": inputs["input_ids"].numel() / elapsed,
            "train/loss": loss,
            # "train/lr": self.optimizers().param_groups[0]["lr"],
            **memory_metrics,
        }

        self.wandb_log(global_state, metrics)
        return loss

    @final
    def validation_step(
        self,
        inputs: DATA_TYPE,
        global_state: GlobalState,
    ):
        # Forward pass through the model
        assert not self.model.training, (
            f"Expected model to be in evaluation mode, got {self.model.training}"
        )

        with elapsed_timer() as timer:
            loss, val_metrics = self.run_validation_step(
                inputs, global_state.global_step
            )
            elapsed = timer()

        metrics = {
            "val/step_time": elapsed,
            "val/loss": loss,
            **val_metrics,
        }
        self.wandb_log(global_state, metrics)
        return metrics

    def configure_optimizers(self) -> OptimizerConfig:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.run_config.lr.peak_lr,
            # foreach=True,
            fused=True,
        )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            partial(
                lr_lambda,
                warmup_steps=self.run_config.lr.warmup_steps,
                end_steps=self.run_config.lr.end_step,
                start_lr=self.run_config.lr.peak_lr,
                end_lr=self.run_config.lr.end_lr,
            ),
        )

        return OptimizerConfig(
            optimizer=optimizer,
            lr_scheduler=LRSchedulerConfig(scheduler=lr_scheduler, interval="step"),
        )
