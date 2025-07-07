from __future__ import annotations

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler
from transformers.modeling_utils import PreTrainedModel
from wandb.sdk.wandb_run import Run
from contextlib import contextmanager
from typing import Iterator

from modeling.config import ExperimentConfig, GlobalState
from lightning.fabric.loggers.logger import _DummyExperiment
from lightning.pytorch.strategies import ModelParallelStrategy
from lightning.pytorch.loggers import Logger
from modeling.checkpoints.save import GCSCheckpointCallback
from synapse.utils.logging import configure_logging
from modeling.utils.tmpdir import TmpDirContext
from synapse.elapsed_timer import elapsed_timer
from modeling.data.data_module import BaseDataModule, BaseDataSample
from modeling.modules.base_module import BaseLITModule, BaseModuleConfig

logger = configure_logging(
    __name__,
    #    level=logging.DEBUG
)


class Initializer:
    """
    Initializer class to handle the setup of the modeling application.
    It initializes the experiment configuration, data pack, module, and logger.
    """

    @staticmethod
    def init_wandb(exp_config: ExperimentConfig) -> list[Logger]:
        """
        Initialize a WandbLogger instance with the configuration.
        """
        loggers: list[Logger] = []
        if wandb_config := exp_config.metadata.wandb:
            logger.debug("Initializing WandbLogger with config: %s", wandb_config)
            wandb_logger = WandbLogger(
                project=wandb_config.project,
                name=wandb_config.name,
                save_dir=exp_config.metadata.output_dir,
            )
            wandb_experiment = wandb_logger.experiment
            logger.debug(
                "WandbLogger initialized with experiment: %s", wandb_experiment
            )
            assert isinstance(wandb_experiment, Run) or isinstance(
                wandb_experiment, _DummyExperiment
            ), f"{wandb_experiment=} should be an instance of wandb.sdk.wandb_run.Run"
            if isinstance(wandb_experiment, Run):
                wandb_experiment.config.update(
                    exp_config.model_dump(serialize_as_any=True)
                )
            logger.debug(
                "WandbLogger configuration updated with experiment config",
            )
            loggers.append(wandb_logger)
        return loggers

    @staticmethod
    @contextmanager
    def init_experiment(
        exp_config: ExperimentConfig, global_state: GlobalState
    ) -> Iterator[
        tuple[
            L.Trainer,
            BaseDataModule[BaseDataSample],
            BaseLITModule[PreTrainedModel, BaseDataSample, BaseModuleConfig],
        ]
    ]:
        """
        Initialize the experiment configuration from a given path.
        """
        global_timer = elapsed_timer("Experiment.Global").__enter__()
        tmpdir_context = TmpDirContext().__enter__()

        loggers = Initializer.init_wandb(exp_config)
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb

        strategy = ModelParallelStrategy(  # <- uses FSDP2 under the hood
            data_parallel_size=exp_config.run.distributed.devices_per_node,  # shard across all 8 GPUs
            tensor_parallel_size=1,  # (= no TP)  just pure FSDP2
            # save_distributed_checkpoint=True,  # write one shard per rank
        )
        logger.debug("Initializing trainer:")
        profiler = PyTorchProfiler(dirpath="output/profiler", emit_nvtx=True)

        trainer = L.Trainer(
            max_epochs=exp_config.run.num_epochs,
            val_check_interval=exp_config.run.validation_every_n_steps
            if exp_config.run.validation_every_n_steps > 0
            else None,
            limit_val_batches=1,
            # Distributed training configuration
            limit_train_batches=exp_config.run.num_epochs
            * exp_config.run.steps_per_epoch,
            accelerator=exp_config.run.accelerator,
            devices=exp_config.run.distributed.devices_per_node,
            num_nodes=exp_config.run.distributed.num_nodes,
            profiler=profiler,
            # Precision and parallel
            strategy=strategy,
            precision=exp_config.run.lightning_precision,
            # Logging and checkpointing
            logger=loggers,
            callbacks=(
                [
                    GCSCheckpointCallback(
                        exp_config=exp_config,
                    )
                ]
                if exp_config.metadata.checkpoint
                else []
            ),
            enable_checkpointing=False,
            default_root_dir=exp_config.metadata.output_dir,
            log_every_n_steps=1,
        )

        try:
            with elapsed_timer("Experiment.Init") as trainer_timer:
                datapack = exp_config.datapack.create_datapack(exp_config)
                assert tmpdir_context.tmpdir is not None
                lit_module = exp_config.module.create_module(
                    exp_config.run, tmpdir_context.tmpdir, global_state
                )

            trainer_timer.print_timing_tree(logger)
            yield trainer, datapack, lit_module
        finally:
            # Clean up temporary directory
            tmpdir_context.__exit__(None, None, None)
            global_timer.__exit__(None, None, None)
            global_timer.print_timing_tree(logger)
