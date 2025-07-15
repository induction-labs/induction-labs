from __future__ import annotations

from wandb.sdk.wandb_run import Run
from contextlib import asynccontextmanager
from typing import AsyncIterator

from modeling.config import ExperimentConfig, UnifiedExperimentConfig
from synapse.utils.logging import configure_logging
from modeling.utils.tmpdir import TmpDirContext
from synapse.elapsed_timer import elapsed_timer
from modeling.modules.base_module import (
    RuntimeConfig,
)
from modeling.config.distributed import DistributedConfig, InstanceConfig
import wandb
from pydantic import AnyUrl
from typing import Optional
from datetime import UTC, datetime
from pathlib import Path
import torch
from dataclasses import dataclass

import secrets
import string

import os
import random
import numpy as np
from modeling.distributed.ray_head import initialize_ray_head

from ray.util.placement_group import (
    placement_group,
)
from modeling.actor import ExperimentActor, ActorArgs
from modeling.utils.typed_remote import RemoteArgs
import asyncio
import logging

logger = configure_logging(__name__, level=logging.DEBUG)


def gen_id(length: int = 8) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


NUM_ACTOR_CPUS = 2.0


def node_resource_str(node_id: int) -> str:
    """
    Generate a resource string for a node based on its ID.
    This is used to specify resources for Ray actors.
    """
    return f"node_{node_id}"


def get_ray_pg_resources(
    distributed_config: DistributedConfig,
) -> list[dict[str, float]]:
    """
    Get the resources for the Ray placement group.
    """
    #
    return [
        {"CPU": NUM_ACTOR_CPUS, "GPU": 1.0, node_resource_str(node_rank): 1.0}
        for device_rank in range(distributed_config.devices_per_node)
        for node_rank in range(distributed_config.num_nodes)
    ]


@dataclass
class RayActors:
    actors: list[list[ExperimentActor]]


@dataclass
class ManagerState:
    """
    Global state for the module, used to store shared information across different parts of the module.
    This can include things like the current step, global loss, etc.
    """

    global_step: int
    tmp_dir: Path
    generator: "torch.Generator"
    ray_host: AnyUrl
    actors: RayActors
    wandb: Optional["Run"]


@dataclass
class ExperimentManager:
    """
    Initializer class to handle the setup of the modeling application.
    It initializes the experiment configuration, data pack, module, and logger.

    This class is not meant to be instantiated directly, only through the context manager interface.
    """

    exp_config: UnifiedExperimentConfig

    state: ManagerState

    # module: BaseLITModule[PreTrainedModel, BaseDataSample, BaseModuleConfig]
    # datapack: BaseDataModule[BaseDataSample]

    # # TODO: Make this async
    async def run(self):
        """
        Run the experiment instance.
        This method is a placeholder for the actual run logic.
        """
        # # Setup data and model

        # self.datapack.setup("fit")
        # self.module.configure_model(self.state.mesh)

        # # Get training dataloader
        # train_dataloader = self.datapack.train_dataloader(
        #     generator=self.state.generator
        # )
        # validation_dataloader = iter(
        #     self.datapack.val_dataloader(generator=self.state.generator)
        # )

        # # Configure optimizer and scheduler
        # optimizer_config = self.module.configure_optimizers()
        # optimizer = optimizer_config.optimizer
        # lr_scheduler = optimizer_config.lr_scheduler.scheduler

        # # Training loop
        # num_epochs = self.exp_config.run.num_epochs
        # steps_per_epoch = self.exp_config.run.steps_per_epoch
        # clipped_grad_norm = float("inf")

        # logger.info(
        #     f"Starting training: {num_epochs} epochs, {steps_per_epoch} steps per epoch"
        # )
        # with profiler_context(
        #     self.exp_config,
        #     self.instance_config,
        # ) as prof:
        #     pbar = tqdm.tqdm(
        #         enumerate(train_dataloader), desc="Training", total=steps_per_epoch
        #     )
        #     for step, batch in pbar:
        #         with elapsed_timer("training_step"):
        #             device = self.module.instance_config.device
        #             assert isinstance(batch, BaseDataSample), f"{batch=}"
        #             if step >= steps_per_epoch:
        #                 break

        #             # Move batch to device
        #             batch = batch.to_device(device)
        #             # Zero gradients
        #             optimizer.zero_grad(set_to_none=True)
        #             # Note: Need to call this or activation checkpointing won't work
        #             # And it will silently fail
        #             self.module.model.train()

        #             # Forward pass
        #             # with torch.profiler.record_function("training_step"):
        #             loss = self.module.training_step(batch, global_state=self.state)

        #             # Backward pass
        #             # with torch.profiler.record_function("backward"):
        #             loss.backward()
        #             real_grad_norm = clip_grad_norm_(
        #                 self.module.model.parameters(), max_norm=clipped_grad_norm
        #             )

        #             # Update weights
        #             # with torch.profiler.record_function("optimizer_step"):
        #             optimizer.step()
        #             lr_scheduler.step()

        #             optimizer.zero_grad(set_to_none=True)

        #             if (
        #                 self.exp_config.run.validation_every_n_steps > 0
        #                 and (self.state.global_step)
        #                 % self.exp_config.run.validation_every_n_steps
        #                 == 0
        #             ):
        #                 logger.debug(
        #                     f"Running validation at step {self.state.global_step}"
        #                 )
        #                 self.module.model.eval()
        #                 with torch.no_grad():
        #                     val_batch = next(validation_dataloader)
        #                     assert isinstance(val_batch, BaseDataSample), (
        #                         f"{val_batch=}"
        #                     )
        #                     val_batch = val_batch.to_device(device)
        #                     # Forward pass for validation
        #                     self.module.validation_step(
        #                         val_batch, global_state=self.state
        #                     )

        #             self.module.wandb_log(
        #                 self.state,
        #                 {
        #                     "lr": lr_scheduler.get_last_lr()[0],
        #                     "real_grad_norm": real_grad_norm.item(),
        #                     "clipped_grad_norm": clipped_grad_norm,
        #                 },
        #                 commit=True,
        #             )
        #             # logger.info(
        #             #     {
        #             #         "lr": lr_scheduler.get_last_lr()[0],
        #             #         "real_grad_norm": real_grad_norm.item(),
        #             #         "clipped_grad_norm": clipped_grad_norm,
        #             #         "loss": loss.item(),
        #             #         "global_step": self.state.global_step,
        #             #         # "embedding_grad_norm": embedding_grad_norm.item(),
        #             #         # "action_head_grad_norm": action_head_grad_norm.item(),
        #             #     }
        #             # )
        #             self.state.global_step += 1

        #             prof.step()

        #             # Update progress bar with loss value
        #             pbar.set_postfix(loss=f"{loss.item():.4f}")

        #     logger.info(f"Training completed. Total steps: {self.state.global_step}")

    @staticmethod
    def create_runtime_config() -> RuntimeConfig:
        """
        Initialize a WandbLogger instance with the configuration.
        """
        id = gen_id(8)
        return RuntimeConfig(
            id=id,
            start_time=datetime.now(tz=UTC),
        )

    @staticmethod
    def init_wandb(exp_config: UnifiedExperimentConfig) -> Run | None:
        """
        Initialize a WandbLogger instance with the configuration.
        """
        if wandb_config := exp_config.metadata.wandb:
            wandb_run = wandb.init(
                project=wandb_config.project,
                name=wandb_config.name,
                id=exp_config.runtime_config.id,
                dir=exp_config.metadata.output_dir,
                config=exp_config.model_dump(
                    serialize_as_any=True, exclude_defaults=False
                ),
            )
            return wandb_run
        return None

    @staticmethod
    def fix_rng(seed: int) -> torch.Generator:
        os.environ["PYTHONHASHSEED"] = str(
            seed
        )  # make hash-based operations reproducible
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
            ":4096:8"  # for reproducibility in cuBLAS
        )

        # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        random.seed(seed)  # Python RNG
        np.random.seed(seed)  # NumPy RNG
        torch.manual_seed(seed)  # CPU RNG
        torch.cuda.manual_seed(seed)  # current GPU RNG
        torch.cuda.manual_seed_all(seed)  # all-GPU RNGs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # # In PyTorch ≥1.8 you can also do:
        torch.use_deterministic_algorithms(True)
        g = torch.Generator()
        g.manual_seed(seed)
        return g

    @staticmethod
    async def initialize_actors(distributed_config: DistributedConfig) -> RayActors:
        actor_args = [
            [
                (
                    RemoteArgs(
                        num_cpus=NUM_ACTOR_CPUS,
                        resources={node_resource_str(node_rank): 1.0},
                    ),
                    ActorArgs(
                        instance_config=InstanceConfig(
                            node_rank=node_rank,
                            device_rank=device_rank,
                        )
                    ),
                )
                for device_rank in range(distributed_config.devices_per_node)
            ]
            for node_rank in range(distributed_config.num_nodes)
        ]
        actor_promises = [
            [
                ExperimentActor.create(args=actor_args, remote_args=remote_args)
                for (remote_args, actor_args) in per_node
            ]
            for per_node in actor_args
        ]
        actors = await asyncio.gather(
            *[asyncio.gather(*per_node) for per_node in actor_promises]
        )
        return RayActors(actors=actors)

    @staticmethod
    @asynccontextmanager
    async def init_experiment(
        exp_config: ExperimentConfig,
    ) -> AsyncIterator[ExperimentManager]:
        """
        Initialize the experiment configuration from a given path.
        """
        global_timer = elapsed_timer("Experiment.Run").__enter__()
        runtime_config = ExperimentManager.create_runtime_config()
        unified_config = UnifiedExperimentConfig(
            runtime_config=runtime_config,
            datapack=exp_config.datapack,
            module=exp_config.module,
            run=exp_config.run,
            metadata=exp_config.metadata,
        )
        generator = ExperimentManager.fix_rng(unified_config.run.seed)
        logger.info("starting ray stuffs")

        with (
            elapsed_timer("Experiment.Init") as trainer_timer,
            TmpDirContext() as (_, tmp_dir),
            initialize_ray_head() as ray_host,
        ):
            logger.debug(f"Ray head node initialized at {ray_host}")
            wandb_run = ExperimentManager.init_wandb(unified_config)

            pg_resources = get_ray_pg_resources(
                unified_config.run.distributed,
            )
            # Strategy doesn't matter here bc we are running 1 process per device.
            pg = placement_group(pg_resources, strategy="PACK")
            logger.debug(f"Placement group created: {pg}")
            # Wait here for worker processes to start
            with elapsed_timer("Experiment.WaitForPlacementGroup"):
                await pg.ready()
            logger.debug("Placement group is ready.")
            with elapsed_timer("Experiment.InitActors"):
                actors = await ExperimentManager.initialize_actors(
                    unified_config.run.distributed
                )
            logger.debug(f"Actors initialized: {actors}")

            manager_state = ManagerState(
                global_step=0,
                tmp_dir=tmp_dir,
                generator=generator,
                wandb=wandb_run,
                ray_host=ray_host,
                actors=actors,
            )
            trainer_timer.print_timing_tree(logger)

            try:
                yield ExperimentManager(
                    exp_config=unified_config,
                    state=manager_state,
                )
            finally:
                # Clean up temporary directory
                if manager_state.wandb is not None:
                    manager_state.wandb.finish()
                global_timer.__exit__(None, None, None)
                global_timer.print_timing_tree(logger)
