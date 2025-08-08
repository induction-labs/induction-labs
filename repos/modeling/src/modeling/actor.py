from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch
from pydantic import BaseModel
from synapse.elapsed_timer import elapsed_timer
from synapse.utils.logging import configure_logging, logging
from torch.optim import Optimizer
from transformers.modeling_utils import PreTrainedModel

from modeling.callbacks.profiler import DummyProfiler, profile, profiler_context
from modeling.checkpoints.save import save_checkpoint_to_tmpdir
from modeling.config import UnifiedExperimentConfig
from modeling.config.data import BaseDataSample
from modeling.config.distributed import InstanceConfig
from modeling.distributed.distributed import TorchUrl, init_distributed
from modeling.modules.base_module import BaseModuleConfig
from modeling.utils.fix_rng import fix_rng
from modeling.utils.flash_attention_utils import (
    configure_flash_attention,
)
from modeling.utils.typed_remote import (
    BaseActor,
    remote_method,
)

if TYPE_CHECKING:
    from modeling.modules.base_module import BaseLITModule

logger = configure_logging(__name__, level=logging.DEBUG)


@dataclass
class InstanceState:
    """
    Global state for the module, used to store shared information across different parts of the module.
    This can include things like the current step, global loss, etc.
    """

    # global_step: int
    mesh: "torch.distributed.device_mesh.DeviceMesh"
    generator: "torch.Generator"
    module: "BaseLITModule[PreTrainedModel, Any, BaseModuleConfig]"
    profile_context: "AbstractContextManager[DummyProfiler | profile] | None" = None
    profiler: Optional["profile| DummyProfiler"] = None

    _optimizer: Optional["Optimizer"] = None
    _lr_scheduler: Optional["torch.optim.lr_scheduler.LRScheduler"] = None

    @property
    def optimizer(self) -> Optimizer:
        assert self._optimizer is not None, "Optimizer has not been set."
        return self._optimizer

    @property
    def lr_scheduler(self) -> "torch.optim.lr_scheduler.LRScheduler":
        assert self._lr_scheduler is not None, "LR Scheduler has not been set."
        return self._lr_scheduler


class ActorArgs(BaseModel):
    instance_config: InstanceConfig
    experiment_config: UnifiedExperimentConfig


class ExperimentActor(BaseActor[ActorArgs]):
    """
    An example actor that can be used to run experiments in a distributed manner.
    """

    state: InstanceState
    distributed_context: AbstractContextManager
    next_sample: BaseDataSample | None = None

    @property
    def instance_config(self) -> InstanceConfig:
        """
        Return the instance configuration for this actor.
        """
        return self.args.instance_config

    @property
    def experiment_config(self) -> UnifiedExperimentConfig:
        """
        Return the experiment configuration for this actor.
        """
        return self.args.experiment_config

    @property
    def g(self) -> torch.Generator:
        """
        Return the random number generator for this actor.
        This is useful for reproducibility in experiments.
        """
        return self.state.generator

    @property
    def mesh(self) -> torch.distributed.device_mesh.DeviceMesh:
        """
        Return the device mesh for this actor.
        This is used for distributed training and operations.
        """
        return self.state.mesh

    @property
    def device(self) -> torch.device:
        """
        Return the device for this actor.
        This is typically used for tensor operations.
        """
        return self.instance_config.device

    @property
    def module(self):
        """
        Return the module for this actor.
        This is the main component that performs the training or inference.
        """
        return self.state.module

    @remote_method
    def get_ip(self) -> str:
        """
        Get the IP address of the actor.
        This can be useful for debugging or logging purposes.
        """
        import socket

        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return local_ip

    def __init__(self, args: ActorArgs):
        super().__init__(args)

    @remote_method
    def init_distributed(self, rank0_address: TorchUrl) -> None:
        """
        Initialize distributed training for this actor.
        This method should be called before any distributed operations.
        """
        assert not hasattr(self, "state"), (
            "This method should not be called after the actor has been initialized."
        )
        torch.set_float32_matmul_precision("high")
        if self.args.experiment_config.run.attn_impl.is_flash_attention:
            configure_flash_attention(
                impl=self.args.experiment_config.run.attn_impl,
            )
        generator = fix_rng(self.args.experiment_config.run.seed, device=self.device)

        self.distributed_context = init_distributed(
            self.experiment_config.run.distributed,
            self.instance_config,
            rank0_address=rank0_address,
        )
        device_mesh = self.distributed_context.__enter__()
        logger.info(f"Distributed training initialized for {self.instance_config=}")

        module = self.experiment_config.module.create_module(
            run_config=self.experiment_config.run,
            instance_config=self.instance_config,
        )
        self.state = InstanceState(
            generator=generator,
            mesh=device_mesh,
            module=module,
            profile_context=profiler_context(self.experiment_config),
        )

    @property
    def model_weights_dir(self) -> Path:
        """
        Return the directory where model weights are stored.
        This is typically used for saving and loading model checkpoints.
        """
        return self.experiment_config.runtime_config.model_weights_dir

    @property
    def tmp_ckpt_path(self) -> Path:
        """
        Return the temporary checkpoint path for the actor.
        This is used to store temporary checkpoints during training.
        """
        return self.experiment_config.runtime_config.tmp_dir / "checkpoints"

    @remote_method
    def configure_model(self) -> None:
        """
        Configure the model for the actor.
        """
        assert hasattr(self, "state"), (
            "This method should not be called before the actor has been initialized."
        )
        logger.info("Configuring model ")
        assert self.model_weights_dir.exists(), (
            f"Model weights directory {self.model_weights_dir} does not exist."
        )
        self.state.module.configure_model(
            device_mesh=self.mesh, weights_dir=self.model_weights_dir
        )
        # Need to reinitialize the optimizer and lr_scheduler after model configuration
        # This is necessary because the model configuration might change the parameters
        # TODO: Split state into state and configured_state?
        self.state._optimizer, self.state._lr_scheduler = (
            self.state.module.configure_optimizers()
        )

    @remote_method
    def shutdown(self) -> None:
        """
        Shutdown the actor.
        This method should be called to clean up resources when the actor is no longer needed.
        """

        if hasattr(self, "distributed_context"):
            logger.info(f"Shutting down distributed context {self.instance_config=}")
            self.distributed_context.__exit__(None, None, None)
        else:
            logger.warning(
                f"No distributed context to shut down on {self.instance_config=}"
            )
        if hasattr(self, "state") and self.state.profile_context is not None:
            logger.info(f"Shutting down profiler context {self.instance_config=}")
            self.state.profile_context.__exit__(None, None, None)
        else:
            logger.warning(
                f"No profiler context to shut down on {self.instance_config=}"
            )

    @remote_method
    def health_check(self) -> float:
        """
        Run a health check on the experiment.
        """
        # Implement the logic to run the experiment here
        # for name, param in self.state.module.model.named_parameters():
        #     check_nans(param, f"{name}")
        x = torch.rand(1, generator=self.g, device=self.device).item()
        return x

    @remote_method
    def start_profiler(self) -> None:
        assert self.state.profile_context is not None, (
            "This method should not be called when the profiler is already started."
        )
        profiler = self.state.profile_context.__enter__()
        self.state.profiler = profiler
        logger.info(f"Profiler started for {self.instance_config=}")

    @remote_method
    def stop_profiler(self) -> None:
        """
        Stop the profiler for the actor.
        This method should be called to stop profiling after the training or validation step.
        """
        if self.state.profile_context is not None:
            self.state.profile_context.__exit__(None, None, None)

    @remote_method
    def set_next_sample(self, sample: BaseDataSample) -> None:
        """
        Set the next sample for the actor.
        This method should be called to set the sample that will be used for training or validation.
        """
        assert hasattr(self, "state"), (
            "This method should not be called before the actor has been initialized."
        )
        assert self.next_sample is None, (
            "This method should not be called when the next sample is already set."
        )

        self.next_sample = sample
        # logger.debug(f"Next sample set for {self.instance_config=}: {sample}")

    @remote_method
    def train_step(self) -> dict[str, float]:
        """
        Perform a training step for the actor.
        This method should be called to train the model.
        """
        assert hasattr(self, "state"), (
            "This method should not be called before the actor has been initialized."
        )
        assert self.next_sample is not None, (
            "This method should not be called before the next sample has been set."
        )
        sample = self.next_sample
        self.next_sample = None

        # Forward pass
        # with
        with (
            torch.profiler.record_function("training_step"),
            elapsed_timer(name="remote_train") as timer,
        ):
            sample = sample.to_device(self.device)
            self.module.model.train()
            self.state.optimizer.zero_grad(set_to_none=True)
            with torch.profiler.record_function("forward"):
                loss, metrics = self.module.training_step(sample)
            # logger.info(f"finished forward pass loss: {loss.item()}")

            # Backward pass
            with torch.profiler.record_function("backward"):
                loss.backward()

            # Update weights
            with torch.profiler.record_function("optimizer_step"):
                self.state.optimizer.step()
                self.state.lr_scheduler.step()
            self.state.optimizer.zero_grad(set_to_none=True)

        # Add learning rate to metrics
        metrics["train/learning_rate"] = self.state.optimizer.param_groups[0]["lr"]
        metrics["train/loss"] = loss.item()
        metrics["train/step_time"] = timer.elapsed
        # logger.debug(f"Training step completed with loss: {loss.item()}")
        if self.state.profiler is not None:
            self.state.profiler.step()
        return metrics

    @remote_method
    def validation_step(self, sample: "BaseDataSample") -> dict[str, float]:
        """
        Perform a validation step for the actor.
        This method should be called to validate the model.
        """
        assert hasattr(self, "state"), (
            "This method should not be called before the actor has been initialized."
        )
        with torch.no_grad():
            sample = sample.to_device(self.device)
            self.module.model.eval()
            metrics = self.module.validation_step(sample)

        # Add learning rate to metrics
        # logger.debug(f"Validation step completed with loss: {loss.item()}")
        return metrics

    @remote_method
    def save_checkpoint_to_tmpdir(self, tmpdir: Path) -> None:
        """
        Save the model checkpoint to the specified temporary directory.
        This method should be called to save the model state.
        """
        assert hasattr(self, "state"), (
            "This method should not be called before the actor has been initialized."
        )
        logger.debug(f"Saving checkpoint to {tmpdir=}")
        save_checkpoint_to_tmpdir(
            model=self.state.module.model,
            local_dir=tmpdir,
        )
