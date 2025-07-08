from __future__ import annotations

from transformers.modeling_utils import PreTrainedModel
from wandb.sdk.wandb_run import Run
from contextlib import contextmanager
from typing import Iterator

from modeling.config import ExperimentConfig, GlobalState
from synapse.utils.logging import configure_logging
from modeling.utils.tmpdir import TmpDirContext
from synapse.elapsed_timer import elapsed_timer
from modeling.data.data_module import BaseDataModule, BaseDataSample
from modeling.modules.base_module import (
    BaseLITModule,
    BaseModuleConfig,
    RuntimeConfig,
    InstanceConfig,
)
import wandb
from modeling.callbacks.tensorboard_handler import tensorboard_trace_handler

from datetime import UTC, datetime
from pathlib import Path
import torch
import tqdm
from dataclasses import dataclass

import secrets
import string

from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
)

logger = configure_logging(
    __name__,
    #    level=logging.DEBUG
)


class UnifiedExperimentConfig(ExperimentConfig):
    runtime_config: RuntimeConfig


def gen_id(length: int = 8) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


class DummyProfiler:
    def step(self):
        pass


@dataclass
class ExperimentInstance:
    """
    Initializer class to handle the setup of the modeling application.
    It initializes the experiment configuration, data pack, module, and logger.

    This class is not meant to be instantiated directly, only through the context manager interface.
    """

    exp_config: UnifiedExperimentConfig

    state: GlobalState
    module: BaseLITModule[PreTrainedModel, BaseDataSample, BaseModuleConfig]
    datapack: BaseDataModule[BaseDataSample]

    async def run(self) -> None:
        """
        Run the experiment instance.
        This method is a placeholder for the actual run logic.
        """
        # Setup data and model
        device = self.module.instance_config.device
        self.datapack.setup("fit")
        self.module.configure_model()

        # Get training dataloader
        train_dataloader = self.datapack.train_dataloader()

        # Configure optimizer and scheduler
        optimizer_config = self.module.configure_optimizers()
        optimizer = optimizer_config.optimizer
        lr_scheduler = optimizer_config.lr_scheduler.scheduler

        # Training loop
        num_epochs = self.exp_config.run.num_epochs
        steps_per_epoch = self.exp_config.run.steps_per_epoch

        logger.info(
            f"Starting training: {num_epochs} epochs, {steps_per_epoch} steps per epoch"
        )
        with ExperimentInstance.profiler_context(self.exp_config) as prof:  # type: ignore[assignment]
            pbar = tqdm.tqdm(
                enumerate(train_dataloader), desc="Training", total=steps_per_epoch
            )
            for step, batch in pbar:
                with elapsed_timer("training_step"):
                    assert isinstance(batch, BaseDataSample), f"{batch=}"
                    if step >= steps_per_epoch:
                        break

                    # Move batch to device
                    batch = batch.to_device(device)
                    # Zero gradients
                    optimizer.zero_grad(set_to_none=True)

                    # Forward pass
                    with torch.profiler.record_function("training_step"):
                        loss = self.module.training_step(batch, global_state=self.state)

                    # Backward pass
                    with torch.profiler.record_function("backward"):
                        loss.backward()

                    # Update weights
                    with torch.profiler.record_function("optimizer_step"):
                        optimizer.step()
                        lr_scheduler.step()

                    self.state.global_step += 1

                    prof.step()

                    # Update progress bar with loss value
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}", step=self.state.global_step
                    )

            logger.info(f"Training completed. Total steps: {self.state.global_step}")

    @staticmethod
    def get_instance_config() -> InstanceConfig:
        """
        Get the instance configuration for the experiment.
        This is used to configure the instance-specific settings.
        """
        # TODO: Fix
        # local_rank = int(os.environ["LOCAL_RANK"])
        # node_rank = int(os.environ["GROUP_RANK"])
        local_rank = node_rank = 0
        return InstanceConfig(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            node_rank=node_rank,
            device_rank=local_rank,
        )

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
    def init_wandb(
        exp_config: UnifiedExperimentConfig,
    ) -> Run | None:
        """
        Initialize a WandbLogger instance with the configuration.
        """
        if wandb_config := exp_config.metadata.wandb:
            wandb_run = wandb.init(
                project=wandb_config.project,
                name=wandb_config.name,
                id=exp_config.runtime_config.id,
                dir=exp_config.metadata.output_dir,
                config=exp_config.model_dump(serialize_as_any=True),
            )
            return wandb_run
        return None

    @staticmethod
    def init_global_state(
        exp_config: UnifiedExperimentConfig,
    ) -> GlobalState:
        """
        Initialize a WandbLogger instance with the configuration.
        """
        wandb_run = ExperimentInstance.init_wandb(exp_config)
        return GlobalState(
            global_step=0,
            wandb=wandb_run,
        )

    @staticmethod
    def do_random_torch_things() -> None:
        """
        Perform random torch operations to ensure that the torch library is initialized.
        This is useful for avoiding issues with lazy initialization in distributed settings.
        """
        # This is a hack to ensure that torch is initialized
        # and that the device mesh is created correctly
        torch._dynamo.config.capture_scalar_outputs = True

        torch.set_float32_matmul_precision("high")

    # TODO: Make this a callback instead of global
    @staticmethod
    @contextmanager
    def profiler_context(
        config: UnifiedExperimentConfig,
    ) -> Iterator[DummyProfiler | profile]:
        # TODO: Wrap profiler in a class instead of union
        """
        Context manager to handle profiling during the experiment.
        This is a placeholder for actual profiling logic.
        """
        try:
            if config.run.profile is None:
                yield DummyProfiler()  # Replace with actual profiling logic
            else:
                profile_dir = config.metadata.output_dir / "profiler"
                profile_schedule = schedule(
                    wait=config.run.profile.wait,
                    warmup=config.run.profile.warmup,
                    active=config.run.profile.active,
                    repeat=config.run.profile.repeat,
                )
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=profile_schedule,
                    on_trace_ready=tensorboard_trace_handler(
                        profile_dir,  # type: ignore[arg-type]
                    ),
                    record_shapes=True,
                    with_stack=True,
                    with_flops=True,
                ) as prof:
                    # prof.start()
                    yield prof
                    # prof.stop()
        except Exception as e:
            logger.error(f"Profiling error: {e}")
            raise e
        finally:
            pass

    @staticmethod
    @contextmanager
    def init_experiment(
        exp_config: ExperimentConfig,
    ) -> Iterator[ExperimentInstance]:
        """
        Initialize the experiment configuration from a given path.
        """
        # We need to enter manually because we need to exit to access timing tree
        global_timer = elapsed_timer("Experiment.Run").__enter__()

        with (
            TmpDirContext() as (tmpdir_context, tmp_dir),
        ):
            ExperimentInstance.do_random_torch_things()
            # TODO: torch.distributed.init_process_group()
            runtime_config = ExperimentInstance.create_runtime_config(tmp_dir)
            instance_config = ExperimentInstance.get_instance_config()
            unified_config = UnifiedExperimentConfig(
                runtime_config=runtime_config,
                datapack=exp_config.datapack,
                module=exp_config.module,
                run=exp_config.run,
                metadata=exp_config.metadata,
            )
            global_state = ExperimentInstance.init_global_state(unified_config)
            # This is so troll
            # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb

            # strategy = ModelParallelStrategy(  # <- uses FSDP2 under the hood
            #     data_parallel_size=exp_config.run.distributed.devices_per_node,  # shard across all 8 GPUs
            #     tensor_parallel_size=1,  # (= no TP)  just pure FSDP2
            #     # save_distributed_checkpoint=True,  # write one shard per rank
            # )
            logger.debug("Initializing trainer:")
            # profiler = PyTorchProfiler(dirpath="output/profiler", emit_nvtx=True)

            # trainer = L.Trainer(
            #     max_epochs=exp_config.run.num_epochs,
            #     val_check_interval=exp_config.run.validation_every_n_steps
            #     if exp_config.run.validation_every_n_steps > 0
            #     else None,
            #     limit_val_batches=1,
            #     # Distributed training configuration
            #     limit_train_batches=exp_config.run.num_epochs
            #     * exp_config.run.steps_per_epoch,
            #     accelerator=exp_config.run.accelerator,
            #     devices=exp_config.run.distributed.devices_per_node,
            #     num_nodes=exp_config.run.distributed.num_nodes,
            #     profiler=profiler,
            #     # Precision and parallel
            #     strategy=strategy,
            #     precision=exp_config.run.lightning_precision,
            #     # Logging and checkpointing
            #     # logger=loggers,
            #     callbacks=(
            #         [
            #             GCSCheckpointCallback(
            #                 exp_config=exp_config,
            #             )
            #         ]
            #         if exp_config.metadata.checkpoint
            #         else []
            #     ),
            #     enable_checkpointing=False,
            #     default_root_dir=exp_config.metadata.output_dir,
            #     log_every_n_steps=1,
            # )

            try:
                with elapsed_timer("Experiment.Init") as trainer_timer:
                    datapack = unified_config.datapack.create_datapack(unified_config)
                    assert tmpdir_context.tmpdir is not None
                    lit_module = unified_config.module.create_module(
                        unified_config.run,
                        unified_config.runtime_config,
                        instance_config=instance_config,
                    )

                trainer_timer.print_timing_tree(logger)
                yield ExperimentInstance(
                    exp_config=unified_config,
                    state=global_state,
                    module=lit_module,
                    datapack=datapack,
                )
            finally:
                # Clean up temporary directory
                if global_state.wandb is not None:
                    global_state.wandb.finish()
                global_timer.__exit__(None, None, None)
                global_timer.print_timing_tree(logger)
