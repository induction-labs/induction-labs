from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Never, cast

import torch
import tqdm
from ray.util.placement_group import (
    PlacementGroup,
    placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from synapse.elapsed_timer import elapsed_timer
from synapse.utils.logging import LOCAL_RANK, NODE_RANK, configure_logging
from torch.utils.data import DataLoader

import wandb
from modeling.actor import ActorArgs, ExperimentActor
from modeling.checkpoints.save import upload_to_gcs
from modeling.config import (
    ExperimentConfig,
    RuntimeConfig,
    UnifiedExperimentConfig,
)
from modeling.config.data import (
    BaseDataSample,
    BaseDataset,
    DataSample,
    SampleWithMetadata,
)
from modeling.config.distributed import DistributedConfig, InstanceConfig
from modeling.distributed.distributed import TorchUrl
from modeling.distributed.ray_head import RayUrl, initialize_ray_head
from modeling.modules.base_module import BaseLITModule, BaseModuleConfig
from modeling.utils.fix_rng import fix_rng
from modeling.utils.gen_id import gen_id
from modeling.utils.max_timeout import max_timeout
from modeling.utils.tmpdir import TmpDirContext
from modeling.utils.typed_remote import RemoteArgs
from wandb.sdk.wandb_run import Run

logger = configure_logging(__name__, level=logging.DEBUG)


NUM_ACTOR_CPUS = 2.0


def node_resource_str(node_id: int) -> str:
    """
    Generate a resource string for a node based on its ID.
    This is used to specify resources for Ray actors.
    """
    return f"Node{node_id}"


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

    @property
    def all_actors(self) -> Iterator[ExperimentActor]:
        """
        Flatten the list of actors across all nodes.
        """
        for actor, _ in self.all_actors_instances:
            yield actor

    @property
    def all_actors_instances(self) -> Iterator[tuple[ExperimentActor, InstanceConfig]]:
        """
        Flatten the list of actors across all nodes, yielding each actor with its instance config.
        """
        for node_index, node_actors in enumerate(self.actors):
            for device_index, actor in enumerate(node_actors):
                # TODO: Tie InstanceConfig to the actor
                yield (
                    actor,
                    InstanceConfig(
                        node_rank=node_index,
                        device_rank=device_index,
                    ),
                )

    @property
    def rank0(self) -> ExperimentActor:
        """
        Get the rank 0 actor (the first actor in the first node).
        This is typically used for logging or other global operations.
        """
        return self.actors[0][0]

    def __len__(self) -> int:
        """
        Return the total number of actors across all nodes.
        """
        return sum(len(node_actors) for node_actors in self.actors)


@dataclass
class ManagerState:
    """
    Global state for the module, used to store shared information across different parts of the module.
    This can include things like the current step, global loss, etc.
    """

    global_step: int
    tmp_dir: Path
    generator: torch.Generator
    ray_host: RayUrl
    actors: RayActors
    train_iter: Iterator[list[SampleWithMetadata[BaseDataSample]]]
    validation_iter: Iterator[list[SampleWithMetadata[BaseDataSample]]]
    wandb: Run | None


def cycle(iterable):
    while True:
        yield from iterable


@dataclass
class ExperimentManager:
    """
    Initializer class to handle the setup of the modeling application.
    It initializes the experiment configuration, data pack, module, and logger.

    This class is not meant to be instantiated directly, only through the context manager interface.
    """

    exp_config: UnifiedExperimentConfig

    state: ManagerState

    def wandb_log(
        self, metrics: dict[str, float], step: int | None = None, commit=False
    ):
        """
        Log metrics to Weights & Biases.
        This is a convenience method to log metrics during training.
        """
        if self.state.wandb is not None:
            self.state.wandb.log(metrics, step=step, commit=commit)

    async def save_checkpoint(self, suffix: str):
        """
        Save a checkpoint of the experiment.
        """
        assert self.exp_config.checkpoint_path is not None, (
            "Checkpointing is not enabled in the experiment configuration."
        )
        local_dir = self.state.tmp_dir / "checkpoints" / suffix

        promises = [
            actor.save_checkpoint_to_tmpdir.remote(tmpdir=local_dir)
            for actor in self.state.actors.all_actors
        ]
        await asyncio.gather(*promises)

        upload_to_gcs(
            local_dir=local_dir,
            gcs_bucket=self.exp_config.checkpoint_path.bucket_and_path[0],
            gcs_prefix=self.exp_config.checkpoint_path.bucket_and_path[1] / suffix,
        )

    # # TODO: Make this async
    async def run(self):
        """
        Run the experiment instance.
        This method is a placeholder for the actual run logic.
        """

        # # Get training dataloader
        # TODO: Don't use torch dataloader class rewrite manually

        module_cls = cast(type[BaseLITModule], self.exp_config.module.module_cls())

        train_iter = self.state.train_iter
        validation_iter = self.state.validation_iter

        # # Training loop
        num_steps = self.exp_config.run.num_steps
        if (c := self.exp_config.metadata.checkpoint) and c.checkpoint_first_step:
            # Save the checkpoint to GCS
            logger.info(f"Saving checkpoint at step 0 to {c.checkpoint_prefix}")
            await self.save_checkpoint(suffix="step_0")

        logger.info(f"Starting training: {num_steps} steps, ")
        pbar = tqdm.tqdm(range(num_steps), desc="Training", total=num_steps)
        for step in pbar:
            if (
                val_steps := self.exp_config.run.validation_every_n_steps
            ) >= 1 and step % val_steps == 0:
                with elapsed_timer("validation_step") as validation_timer:
                    logger.debug(f"Running validation step at {step=}")
                    validation_batch = next(validation_iter)
                    indices = [data.indices for data in validation_batch]
                    logger.info(f"Validation step {step}: {indices=} indices")
                    all_validation_metrics = await asyncio.gather(
                        *[
                            actor.validation_step.remote(data.sample)
                            for actor, data in zip(
                                self.state.actors.all_actors,
                                validation_batch,
                                strict=True,
                            )
                        ]
                    )
                    # Reduce metrics to get mean
                    validation_metrics = module_cls.validation_wandb_metrics(
                        all_validation_metrics, global_step=step
                    )
                validation_metrics["validation/head_time"] = validation_timer.elapsed
                self.wandb_log(validation_metrics, step=step)
            with elapsed_timer("training_step") as training_timer:
                with elapsed_timer("wait_train_data") as train_data_timer:
                    batch = next(train_iter)
                assert isinstance(batch, list), f"{batch=}"
                assert len(batch) == len(self.state.actors), (
                    f"{len(batch)=} != {len(self.state.actors)=}"
                )
                indices = [data.indices for data in batch]
                # logger.info(f"Training step {step}: {indices=} indices")

                all_train_metrics = await asyncio.gather(
                    *[
                        actor.train_step.remote(data.sample)
                        for actor, data in zip(
                            self.state.actors.all_actors, batch, strict=True
                        )
                    ]
                )
                # Reduce metrics to get mean

                train_metrics = module_cls.training_wandb_metrics(all_train_metrics)
            train_metrics["train/head_time"] = training_timer.elapsed
            train_metrics["train/data_time"] = train_data_timer.elapsed
            self.wandb_log(train_metrics, step=step)
            # logger.info(train_metrics)
            loss = train_metrics.get("train/loss")
            # TODO: Make this a callback
            if (c := self.exp_config.metadata.checkpoint) and c.should_checkpoint(step):
                with elapsed_timer("save_checkpoint") as save_timer:
                    # Save the checkpoint to GCS
                    logger.info(
                        f"Saving checkpoint at step {step} to {self.exp_config.checkpoint_path}"
                    )
                    await self.save_checkpoint(suffix=f"step_{step}")
                self.wandb_log(
                    {
                        "checkpoint/save_time": save_timer.elapsed,
                    }
                )
            self.wandb_log({}, step=step, commit=True)
            pbar.set_postfix(loss=f"{loss:.4f}")

        step = num_steps
        if (
            val_steps := self.exp_config.run.validation_every_n_steps
        ) >= 1 and step % val_steps == 0:
            logger.debug(f"Running validation step at {step=}")
            validation_batch = next(validation_iter)
            all_validation_metrics = await asyncio.gather(
                *[
                    actor.validation_step.remote(data.sample)
                    for actor, data in zip(
                        self.state.actors.all_actors, validation_batch, strict=True
                    )
                ]
            )
            # Reduce metrics to get mean
            validation_metrics = module_cls.validation_wandb_metrics(
                all_validation_metrics, global_step=step
            )
            self.wandb_log(validation_metrics, step=step)
        # Post training stuff
        # TODO: Write a finished training cleanup hook
        logger.info("🚀🚀🚀🚀🚀🚀🚀 Run completed 🚀🚀🚀🚀🚀🚀🚀")
        if (c := self.exp_config.metadata.checkpoint) and c.checkpoint_last_step:
            # Save the checkpoint to GCS
            logger.info(f"Saving checkpoint at last step to {c.checkpoint_prefix}")
            await self.save_checkpoint(suffix="step_-1")

    @staticmethod
    def create_runtime_config(tmp_dir: Path) -> RuntimeConfig:
        """
        Initialize a WandbLogger instance with the configuration.
        """
        id = gen_id(8)
        return RuntimeConfig(
            id=id,
            start_time=datetime.now(tz=UTC),
            tmp_dir=tmp_dir,
        )

    @staticmethod
    def make_dataloader(
        dataset: BaseDataset[DataSample, Never],
        full_config: ExperimentConfig[DataSample],
        generator: torch.Generator,
    ) -> tuple[Iterator[list[SampleWithMetadata[DataSample]]], dict]:
        """
        Create a DataLoader for the training dataset.
        This method is used to create a DataLoader for the training dataset with the specified batch size and collate function.
        Returns a tuple of (dataloader, config_dict).
        """
        # seed = full_config.run.seed
        # generator = torch.Generator(device="cpu").manual_seed(seed)

        # see https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        data_loader = DataLoader(
            dataset,
            batch_size=full_config.run.batch_size,
            shuffle=True,
            drop_last=True,
            generator=generator,
            # TODO: Figure out optimal prefetch factor + num_workers
            prefetch_factor=full_config.run.dataloader_prefetch_factor,
            # Need num_workers!=0 so that this runs in MP mode, so that
            num_workers=full_config.run.dataloader_num_workers,
            persistent_workers=True,
            collate_fn=dataset.collate_fn(
                batch_size=full_config.run.batch_size,
                world_size=full_config.run.distributed.world_size,
            ),
        )
        # We need to cast here because DataLoader should be parameterized with the return type of collate_fn
        # but it currently isn't due to limitations in mypy https://github.com/python/mypy/issues/3737
        typed_data_loader = cast(
            DataLoader[list[SampleWithMetadata[DataSample]]], data_loader
        )

        # Get configuration information
        batch_size = data_loader.batch_size
        dataset_size = len(dataset) if hasattr(dataset, "__len__") else "unknown"
        num_steps = len(data_loader)

        # Get generator seed if available

        sampler_name = (
            data_loader.sampler.__class__.__name__ if data_loader.sampler else "None"
        )

        config = {
            "batch_size": batch_size,
            "total_samples": dataset_size,
            "steps_per_epoch": num_steps,
            "generator_seed": generator.initial_seed(),
            "shuffle": sampler_name,
            "drop_last": data_loader.drop_last,
            "num_workers": data_loader.num_workers,
        }

        # Note that we return the ITERATOR of the dataloader not just the dataloader itself
        # This is to start prefetching data - the MP prefetching only starts when the iterator is created,
        # not when the DataLoader is created.

        iterator = cast(
            Iterator[list[SampleWithMetadata[DataSample]]],
            iter(cycle(typed_data_loader)),
        )

        return iterator, config

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
    @asynccontextmanager
    async def initialize_actors(
        experiment_config: UnifiedExperimentConfig, pg: PlacementGroup
    ) -> AsyncIterator[RayActors]:
        distributed_config = experiment_config.run.distributed

        actor_args = [
            [
                (
                    RemoteArgs(
                        num_cpus=NUM_ACTOR_CPUS,
                        num_gpus=1.0,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                        ),
                        resources={node_resource_str(node_rank): 1.0},
                    ),
                    ActorArgs(
                        instance_config=InstanceConfig(
                            node_rank=node_rank,
                            device_rank=device_rank,
                        ),
                        experiment_config=experiment_config,
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
        try:
            yield RayActors(actors=actors)
        finally:
            # Shutdown all actors
            logger.debug("Shutting down all actors...")

            # Wrap this in a try-except with a timeout to ensure we don't hang forever
            shutdown_promises = [
                actor.shutdown.remote() for per_node in actors for actor in per_node
            ]
            await max_timeout(
                asyncio.gather(*shutdown_promises),
                timeout=timedelta(seconds=60),
                err_msg="Timed out while shutting down actors",
            )

            logger.debug("All actors have been shut down.")

    @staticmethod
    def ray_head_resources(distributed_config: DistributedConfig):
        # Ray doesn't let you do `ray.init(resources={})` if connecting to an existing cluster,
        # so we just compute how much we need here instead of passing it to `ray.init()`.
        assert distributed_config.num_nodes == 1, (
            "This method is only for initializing a single node Ray head worker."
        )
        return {
            node_resource_str(0): float(distributed_config.devices_per_node),
        }

    @staticmethod
    def download_weights(unified_config: UnifiedExperimentConfig) -> None:
        """
        Download the model weights for the module.
        This method should be called after the actor has been initialized.
        This should only be called on global_rank 0
        """
        os.makedirs(unified_config.runtime_config.model_weights_dir, exist_ok=False)
        module_cls = unified_config.module.module_cls()
        assert isinstance(unified_config.module, BaseModuleConfig)
        module_cls.download_weights(
            module_config=unified_config.module,
            tmpdir=unified_config.runtime_config.model_weights_dir,
        )

    @staticmethod
    @asynccontextmanager
    async def init_experiment(
        exp_config: ExperimentConfig, ray_head_worker=False
    ) -> AsyncIterator[ExperimentManager]:
        """
        Initialize the experiment configuration from a given path.
        """
        logger.info("Initializing experiment...")
        global_timer = elapsed_timer("Experiment.Run").__enter__()
        (tmpdir_context, tmp_dir) = TmpDirContext().__enter__()

        runtime_config = ExperimentManager.create_runtime_config(tmp_dir)
        unified_config = UnifiedExperimentConfig(
            runtime_config=runtime_config,
            train_datapack=exp_config.train_datapack,
            validation_datapack=exp_config.validation_datapack,
            module=exp_config.module,
            run=exp_config.run,
            metadata=exp_config.metadata,
        )
        generator = fix_rng(unified_config.run.seed)
        head_resources = (
            ExperimentManager.ray_head_resources(
                unified_config.run.distributed,
            )
            if ray_head_worker
            else None
        )

        with (
            elapsed_timer("Experiment.Init"),
            initialize_ray_head(head_resources) as ray_host,
        ):
            wandb_run = ExperimentManager.init_wandb(unified_config)
            pg_resources = get_ray_pg_resources(
                unified_config.run.distributed,
            )
            # Strategy doesn't matter here bc we are running 1 process per device.
            pg = placement_group(pg_resources, strategy="PACK")
            logger.debug(f"Placement group created: {pg_resources=}")
            # Wait here for worker processes to start
            with elapsed_timer("Experiment.WaitForPlacementGroup"):
                # TODO: Put a timer on this and error if it takes too long
                await max_timeout(
                    pg.ready(),
                    timeout=timedelta(seconds=60),
                    err_msg="Placement group did not become ready in time",
                )
            logger.debug("Placement group is ready.")

            async with ExperimentManager.initialize_actors(
                unified_config, pg
            ) as actors:
                logger.debug(f"Actors created: {len(actors)=}")

                with elapsed_timer("Experiment.SetEnviron"):
                    worker_env = {
                        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
                    }
                    logger.debug(f"Setting worker environment: {worker_env}")
                    await asyncio.gather(
                        *[
                            actor.set_environ.remote(
                                {
                                    NODE_RANK: str(i.node_rank),
                                    LOCAL_RANK: str(i.device_rank),
                                    **worker_env,
                                }
                            )
                            for actor, i in actors.all_actors_instances
                        ]
                    )

                with elapsed_timer("Experiment.InitDistributed"):
                    rank0_host = await actors.rank0.get_ip.remote()
                    rank0_address = TorchUrl.build(
                        host=rank0_host,
                        port=23456,
                        scheme="tcp",
                    )
                    logger.debug(f"Rank 0 address: {rank0_address}")
                    distributed_promise = asyncio.gather(
                        *[
                            actor.init_distributed.remote(rank0_address)
                            for actor in actors.all_actors
                        ]
                    )

                with elapsed_timer(
                    "Experiment.InitDataloaders"
                ) as init_dataloader_timer:
                    # We initialize dataloaders here because it is a head only operation and
                    # init distributed takes a long time.
                    train_dataset, validation_dataset = await asyncio.gather(
                        unified_config.train_datapack._init_dataset(
                            full_config=unified_config,
                        ),
                        unified_config.validation_datapack._init_dataset(
                            full_config=unified_config,
                        ),
                    )
                    train_dataiter, train_config = ExperimentManager.make_dataloader(
                        dataset=train_dataset,
                        full_config=unified_config,
                        generator=generator,
                    )
                    validation_dataiter, validation_config = (
                        ExperimentManager.make_dataloader(
                            dataset=validation_dataset,
                            full_config=unified_config,
                            generator=generator,
                        )
                    )
                logger.debug(
                    f"Dataloaders initialized in {init_dataloader_timer.elapsed:.2f}"
                )
                logger.info(f"Train dataloader config: {train_config}")
                logger.info(f"Validation dataloader config: {validation_config}")

                with elapsed_timer(
                    "Experiment.DownloadWeights"
                ) as download_weights_timer:
                    # Download the model weights to the actors
                    ExperimentManager.download_weights(unified_config)
                logger.debug(
                    f"Model weights downloaded to {unified_config.runtime_config.model_weights_dir} in {download_weights_timer.elapsed:.2f} seconds."
                )

                with elapsed_timer("Experiment.WaitForDistributed") as wait_dist_timer:
                    # Wait for all actors to initialize distributed training
                    await max_timeout(
                        distributed_promise,
                        timeout=timedelta(seconds=60 * 5),
                        err_msg="Distributed training did not initialize in time",
                    )

                logger.debug(
                    f"Distributed training initialized for all actors in {wait_dist_timer.elapsed:.2f} seconds."
                )

                with elapsed_timer("Experiment.ConfigureModel"):
                    await asyncio.gather(
                        *[actor.configure_model.remote() for actor in actors.all_actors]
                    )
                logger.debug("Model configured for all actors.")

                health_checks = await asyncio.gather(
                    *[actor.health_check.remote() for actor in actors.all_actors]
                )
                # Make sure all health check values are the same
                assert all(
                    health_check == health_checks[0] for health_check in health_checks
                ), (
                    f"Health checks returned different values across actors. {health_checks=}"
                )
                logger.debug(f"Health checks completed: {health_checks}")

                manager_state = ManagerState(
                    global_step=0,
                    tmp_dir=tmp_dir,
                    generator=generator,
                    wandb=wandb_run,
                    ray_host=ray_host,
                    actors=actors,
                    train_iter=train_dataiter,
                    validation_iter=validation_dataiter,
                )

                try:
                    yield ExperimentManager(
                        exp_config=unified_config,
                        state=manager_state,
                    )
                finally:
                    tmpdir_context.__exit__(None, None, None)
                    # trainer_timer.print_timing_tree(logger)

                    # Clean up temporary directory
                    if manager_state.wandb is not None:
                        manager_state.wandb.finish()
                    global_timer.__exit__(None, None, None)
                    global_timer.print_timing_tree(logger)
