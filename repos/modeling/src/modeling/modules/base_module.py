from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from glob import glob
from pathlib import Path
from functools import partial

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from modeling.utils.cloud_path import CloudPath
from safetensors import safe_open
import torch
from modeling.config import (
    InstanceConfig,
    ModuleConfig,
    RunConfig,
)
from modeling.utils.elapsed_timer import elapsed_timer
from modeling.utils.get_attn_impl import check_attn_impl
from torch.distributed.device_mesh import DeviceMesh
from torch.optim.adamw import AdamW
from transformers.configuration_utils import PretrainedConfig

from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from transformers.modeling_utils import PreTrainedModel
from typing import Any, Generic, Literal, Optional, TypeVar, cast, final
from synapse.utils.logging import configure_logging
from modeling.checkpoints.load import download_model_checkpoint
from pydantic import BaseModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
from transformers.modeling_utils import no_init_weights
import logging
from modeling.utils.class_property import class_property

logger = configure_logging(
    __file__,
    level=logging.INFO,  # Set to DEBUG for more verbose output
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
        instance_config: InstanceConfig,
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


def get_param_with_parent_module(
    module: torch.nn.Module, name: str
) -> Optional[tuple[torch.nn.Module, torch.nn.Parameter, str]]:
    module_path, _, param_name = name.rpartition(".")

    try:
        mod: torch.nn.Module = module.get_submodule(module_path)
    except AttributeError:
        # Module does not exist
        return None

    if not hasattr(mod, param_name):
        return None

    param = getattr(mod, param_name)

    if not isinstance(param, torch.nn.Parameter):
        # Actually raise error here
        raise AttributeError("`" + param_name + "` is not an nn.Parameter")

    return mod, param, param_name


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

    @final
    def __init__(
        self,
        module_config: CONFIG_TYPE,
        run_config: RunConfig,
        instance_config: InstanceConfig,
    ):
        # self.runtime_config = runtime_config
        self.instance_config = instance_config
        self.module_config = module_config
        self.run_config = run_config
        self.attn_impl = run_config.attn_impl
        self.dtype = run_config.precision.torch_dtype

        check_attn_impl(self.attn_impl)
        logger.debug(f"Using attention implementation: {self.attn_impl}, {self.dtype=}")
        if self.run_config.accelerator == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with init_empty_weights():
            self.model = self.init_model_meta()

    # @abstractmethod
    @final
    def activation_checkpoint_model(self) -> MODEL_TYPE:
        """
        !!! IMPORTANT - use torch.distributed.algorithms._checkpoint.checkpoint_wrapper, NOT HF self.model.gradient_checkpointing_enable()
        !!! HF gradient_checkpointing_enable() does not work with FSDP - once sharded, no activation checkpointing is applied

        !!! JUST KIDDING - checkpoint_wrapper is broken with qwen 25o for some reason - I think it is because
        !!! HF Transformers relies on __call__ and i think that default checkpoint_wrapper only wraps forward so
        !!! there are some bugs here https://github.com/huggingface/transformers/blob/0e1c2817455602d182bd8ebf5fba212e14fb187e/src/transformers/modeling_layers.py#L27


        !!! So HF checkpoint_wrapper by default uses reentrant=True, which is what I think is broken with fsdp.
        !!! For now just going to use HF's gradient_checkpointing_enable() method with reentrant=False which seems to work for now :/
        Abstract method to be implemented by subclasses for enabling activation checkpointing.
        This method should return the model with activation checkpointing enabled.
        """
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        return self.model

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

    @final
    def configure_model(self, device_mesh: DeviceMesh, weights_dir: Path) -> None:
        # We need to ensure that all models are fsdp because that is we use by default
        logger.debug("Configuring model for FSDP sharding...")

        if isinstance(self.model, FSDPModule):
            raise RuntimeError(
                "Model is already sharded with FSDP. This should not happen, please report this issue."
            )
        # TODO: We should probably shard the model first and load sharded weights
        self.model = self.load_weights(weights_dir)

        # Very specific order here:
        # 1. Enable activation checkpointing if configured
        # 2. Shard the model using FSDP
        # 3. Compile the model if configured

        if self.module_config.activation_checkpointing is not None:
            logger.debug("Enabling activation checkpointing...")
            self.model = self.activation_checkpoint_model()  # type: ignore[assignment]
        # # Enable activation checkpointing if configured

        self.model = self.shard_model(
            mp_policy=self.run_config.mp_policy,
            device_mesh=device_mesh,
        )  # type: ignore[assignment]

        assert isinstance(self.model, FSDPModule), (
            f"Expected self.model to be a FullyShardedDataParallel, got {type(self.model)}"
        )
        # logger.debug(f"Sharded model {self.model} with dtype {self.dtype}")
        # NOTE: For now we are just going to shard all params, and `ignore_params` is not supported because
        # sharding in torch is all manual anyways and if we want to support replicate on some params only then
        # we would basically need to rewrite into `torch.distributed.fsdp._fully_shard._fsdp_param.FSDPParam` class
        # the logic of how to handle replicate params and sharded params.

        # We can only compile *after* sharding and gradient checkpointing, otherwise it doesn't trace through.
        if self.module_config.compile is not None:
            self.model = torch.compile(
                self.model,
                mode=self.module_config.compile.mode,
                fullgraph=self.module_config.compile.fullgraph,
            )  # type: ignore[assignment]

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
        should_err = False
        self.model = self.model.to_empty(device=self.device)
        self.model = self.model.to(dtype=self.dtype)

        # PARTIALLY_INITIALIZED = "_PARTIALLY_INITIALIZED"
        # Convert dtype of self.model
        # First move buffers to the correct device

        # # Load the model weights and dispatch them to the appropriate devices
        # for module in self.model.modules():
        #     # This property is set by the HF load_checkpoint_and_dispatch method
        #     if getattr(module, "_is_hf_initialized", False) or getattr(
        #         module, PARTIALLY_INITIALIZED, False
        #     ):
        #         # logger.warning(
        #         #     f"Module {module} was already initialized. "
        #         #     "This may cause issues with training or inference."
        #         # )
        #         setattr(module, "_is_hf_initialized", False)
        #         # should_err = True
        # if should_err:
        #     raise RuntimeError(
        #         "Some modules were already initialized. "
        #         "This may cause issues with training or inference. "
        #         "Please check your model configuration and ensure that the model is not already initialized."
        #     )

        # # print(self.model.state_dict().keys())

        # unused_params: list[str] = []
        # with torch.no_grad():
        #     for shard in sorted(glob(str(tmpdir / "*.safetensors"))):
        #         with safe_open(
        #             shard, framework="pt", device=self.instance_config.device_rank
        #         ) as f:
        #             for name in f.keys():
        #                 assert isinstance(name, str)
        #                 model_name = name.replace("thinker.", "", 1)
        #                 pair = get_param_with_parent_module(self.model, model_name)
        #                 if pair is None:
        #                     unused_params.append(name)
        #                     continue
        #                 mod, param, param_name = pair
        #                 tensor: torch.Tensor = f.get_tensor(name)
        #                 target_dtype = (
        #                     self.dtype
        #                     if isinstance(param, torch.nn.Parameter)
        #                     else torch.float32
        #                 )
        #                 # convert tensor to the correct dtype
        #                 if tensor.dtype != target_dtype:
        #                     tensor = tensor.to(target_dtype)
        #                 param = torch.nn.Parameter(
        #                     torch.empty_like(
        #                         param, device=self.device, dtype=target_dtype
        #                     ),
        #                     requires_grad=param.requires_grad,
        #                 )
        #                 param.data.copy_(tensor)
        #                 # TODO: For now we use hf `_is_hf_initialized` convention.
        #                 setattr(param, "_is_hf_initialized", True)
        #                 setattr(mod, param_name, param)
        #                 setattr(mod, PARTIALLY_INITIALIZED, True)

        # # print(f"{all_params=}")
        # if len(unused_params) > 0:
        #     logger.warning(
        #         f"Some parameters were not used when initializing the model: {len(unused_params)=}"
        #         f"{unused_params[:10]}..."
        #     )

        # # logger.debug(f"{unused_params=}")

        # uninitialized_modules: list[str] = []
        # for name, module in self.model.named_modules():
        #     child_parameters = list(module.named_parameters(recurse=False))
        #     # Check that the module has direct child parameters
        #     if len(child_parameters) == 0:
        #         # If it has no child parameters, don't need to check it
        #         setattr(module, "_is_hf_initialized", True)
        #         continue

        #     # check partially initialized modules
        #     if getattr(module, PARTIALLY_INITIALIZED, False):
        #         # Check that all direct child params are initialized
        #         for name, param in child_parameters:
        #             assert module.device == param.device, (
        #                 f"Module {name} has parameter {param} "
        #             )
        #             if not param.device == self.device:
        #                 raise RuntimeError(
        #                     f"{module=} was partially initialized, uninitialized parameter {name}. "
        #                 )
        #         setattr(module, "_is_hf_initialized", True)

        #     else:
        #         # Here none of the parameters are initialized
        #         for name, param in child_parameters:
        #             param = torch.nn.Parameter(
        #                 torch.empty_like(param, device=self.device, dtype=self.dtype),
        #                 requires_grad=param.requires_grad,
        #             )
        #             setattr(module, name, param)
        #         uninitialized_modules.append(name)

        # if len(uninitialized_modules) > 0:
        #     logger.warning(
        #         f"Some modules were not loaded from weights: {len(uninitialized_modules)=}"
        #         f"{uninitialized_modules[:10]}..."
        #     )
        # logger.debug(f"{uninitialized_modules=}")

        # Now we init rest of the model weights
        self.model.init_weights()

        # # TODO: Either figure out how accelerate load_checkpoint_and_dispatch works in detail
        # # TODO: or just use the HF load_model method

        # Finally check that all modules were initialized
        for name, module in self.model.named_modules():
            # This property is set by the HF load_checkpoint_and_dispatch method
            if not getattr(module, "_is_hf_initialized", False):
                logger.warning(f"Module {name} was not loaded or initialized. ")
                should_err = True
        if should_err:
            raise RuntimeError(
                "Some modules were not initialized. "
                "This may cause issues with training or inference. "
                "Please check your model configuration and ensure that the model is properly initialized."
            )

        self.model.to(
            device=self.device,
        )

        loaded_model = self.model_cls.from_pretrained(
            self.module_config.model_name,
            config=self.model.config,
            torch_dtype=self.dtype,
            device_map={
                "": self.device  # Use the device index for the model
            },  # Ensure model is loaded on the correct device
            attn_implementation=self.attn_impl,
            local_files_only=True,
        )
        # Check the difference between the loaded model and the current model
        for name, param in self.model.named_parameters():
            pair = get_param_with_parent_module(loaded_model, name)
            if pair is None:
                logger.warning(
                    f"Loaded model has parameter {name} that is not in the current model."
                )
                continue

            current_mod, current_param, _ = pair
            # Check dtype and device
            if current_param.dtype != param.dtype:
                logger.warning(
                    f"Parameter {name} in loaded model has dtype {param.dtype}, "
                    f"but current model has dtype {current_param.dtype}."
                )
            if current_param.device != param.device:
                logger.warning(
                    f"Parameter {name} in loaded model is on device {param.device}, "
                    f"but current model is on device {current_param.device}."
                )

            if not torch.equal(param.data, current_param.data):
                logger.warning(
                    f"Parameter {name} in loaded model does not match current model."
                )
                # then copy the data from the loaded model to the current model
                current_param.data.copy_(param.data)

        our_named_buffers = dict(self.model.named_buffers())
        their_named_buffers = dict(loaded_model.named_buffers())

        for k, v in our_named_buffers.items():
            if k not in their_named_buffers:
                logger.warning(f"Buffer {k} is not in the loaded model.")
                continue
            their_v = their_named_buffers[k]
            if v.dtype != their_v.dtype:
                logger.warning(
                    f"Buffer {k} has dtype {v.dtype}, but loaded model has dtype {their_v.dtype}."
                )
            if v.device != their_v.device:
                logger.warning(
                    f"Buffer {k} is on device {v.device}, but loaded model is on device {their_v.device}."
                )
            if not torch.equal(v, their_v):
                logger.warning(f"Buffer {k} does not match loaded model.")
                # print(f"{v=}, {their_v=}")
                # v.copy_(their_v)

        for k, v in their_named_buffers.items():
            if k not in our_named_buffers:
                logger.warning(
                    f"Buffer {k} is in the loaded model, but not in our model."
                )

        return self.model

    # @final
    # def wandb_log(
    #     self, global_state: GlobalState, metrics: dict[str, Any], commit: bool = False
    # ) -> None:
    #     """
    #     Log metrics to Wandb if available.
    #     This method is a wrapper around the logger's log_dict method.
    #     Should not call wandb commit except in trainer.
    #     # TODO: Rewrite with a wandb commit callback thing.
    #     """
    #     # logger.info(metrics)
    #     if global_state.wandb is not None:
    #         global_state.wandb.log(metrics, commit=commit)

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
        assert torch.isfinite(loss).all(), (
            f"Expected loss to be finite, got {loss} with dtype {loss.dtype}"
        )
        return loss

    @final
    def training_step(
        self,
        inputs: DATA_TYPE,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
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

        return loss, metrics

    @final
    def validation_step(
        self,
        inputs: DATA_TYPE,
    ):
        # Forward pass through the model
        assert not self.model.training, (
            f"Expected model to be in evaluation mode, got {self.model.training}"
        )

        with elapsed_timer() as timer:
            loss, val_metrics = self.run_validation_step(inputs, 0)
            elapsed = timer()

        metrics = {
            "val/step_time": elapsed,
            "val/loss": loss,
            **val_metrics,
        }
        return metrics

    def configure_optimizers(self) -> tuple[Optimizer, LRScheduler]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.run_config.lr.peak_lr,
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

        return (optimizer, lr_scheduler)
