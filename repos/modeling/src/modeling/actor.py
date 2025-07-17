from modeling.modules.base_module import BaseModuleConfig, PreTrainedModel
from modeling.utils.typed_remote import (
    remote_method,
    BaseActor,
)
from pydantic import BaseModel
from modeling.config.distributed import InstanceConfig
from modeling.config import UnifiedExperimentConfig
from modeling.utils.fix_rng import fix_rng
from synapse.utils.logging import configure_logging, logging
from contextlib import AbstractContextManager
import torch
from modeling.distributed.distributed import init_distributed, TorchUrl
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
import os
from modeling.config.data import BaseDataSample
from torch.optim import Optimizer
from modeling.utils.check_nans import check_nans

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
    optimizer: "Optimizer"
    lr_scheduler: "torch.optim.lr_scheduler.LRScheduler"


class ActorArgs(BaseModel):
    instance_config: InstanceConfig
    experiment_config: UnifiedExperimentConfig


class ExperimentActor(BaseActor[ActorArgs]):
    """
    An example actor that can be used to run experiments in a distributed manner.
    """

    state: InstanceState
    distributed_context: AbstractContextManager

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
        optimizer, lr_scheduler = module.configure_optimizers()
        self.state = InstanceState(
            # global_step=0,
            generator=generator,
            mesh=device_mesh,
            module=module,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    @property
    def model_weights_dir(self) -> Path:
        """
        Return the directory where model weights are stored.
        This is typically used for saving and loading model checkpoints.
        """
        return self.experiment_config.runtime_config.tmp_dir / "model_weights"

    @remote_method
    def download_weights(self) -> None:
        """
        Download the model weights for the module.
        This method should be called after the actor has been initialized.
        This should only be called on global_rank 0
        """
        assert hasattr(self, "state"), (
            "This method should not be called before the actor has been initialized."
        )
        os.makedirs(self.model_weights_dir, exist_ok=True)
        self.state.module.download_weights(self.model_weights_dir)

    @remote_method
    def configure_model(self) -> None:
        """
        Configure the model for the actor.
        """
        assert hasattr(self, "state"), (
            "This method should not be called before the actor has been initialized."
        )
        os.makedirs(self.model_weights_dir, exist_ok=True)
        self.state.module.configure_model(
            device_mesh=self.mesh, weights_dir=self.model_weights_dir
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

    @remote_method
    def health_check(self) -> float:
        """
        Run a health check on the experiment.
        """
        # Implement the logic to run the experiment here
        for name, param in self.state.module.model.named_parameters():
            check_nans(param, f"{name}")
        x = torch.rand(1, generator=self.g, device=self.device).item()
        return x

    @remote_method
    def train_step(self, sample: "BaseDataSample") -> dict[str, float]:
        """
        Perform a training step for the actor.
        This method should be called to train the model.
        """
        assert hasattr(self, "state"), (
            "This method should not be called before the actor has been initialized."
        )
        sample = sample.to_device(self.device)
        self.module.model.train()
        self.state.optimizer.zero_grad(set_to_none=True)

        # Forward pass
        # with torch.profiler.record_function("training_step"):
        loss, metrics = self.module.training_step(sample)
        # self.wanddblog(metrics)

        # Backward pass
        # with torch.profiler.record_function("backward"):
        loss.backward()

        # Update weights
        # with torch.profiler.record_function("optimizer_step"):
        self.state.optimizer.step()
        self.state.lr_scheduler.step()
        self.state.optimizer.zero_grad(set_to_none=True)

        # Add learning rate to metrics
        metrics["train/learning_rate"] = self.state.optimizer.param_groups[0]["lr"]
        metrics["train/loss"] = loss.item()
        return metrics
