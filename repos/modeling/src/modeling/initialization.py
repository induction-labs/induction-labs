from __future__ import annotations

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from wandb.sdk.wandb_run import Run

from modeling.config import ExperimentConfig
from lightning.fabric.loggers.logger import _DummyExperiment
from lightning.pytorch.strategies import ModelParallelStrategy
from lightning.pytorch.loggers import Logger
from modeling.checkpoints.save import GCSCheckpointCallback


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
            # TODO: For some reason lightnign / wanbd is fucking logging checkpoints, i cant find the setting for this right now but turn it off.
            wandb_logger = WandbLogger(
                project=wandb_config.project,
                name=wandb_config.name,
                save_dir=exp_config.metadata.output_dir,
            )
            wandb_experiment = wandb_logger.experiment
            assert isinstance(wandb_experiment, Run) or isinstance(
                wandb_experiment, _DummyExperiment
            ), f"{wandb_experiment=} should be an instance of wandb.sdk.wandb_run.Run"
            if isinstance(wandb_experiment, Run):
                wandb_experiment.config.update(
                    exp_config.model_dump(serialize_as_any=True)
                )
            loggers.append(wandb_logger)
        return loggers

    @staticmethod
    def init_experiment(
        exp_config: ExperimentConfig,
    ) -> tuple[L.Trainer, L.LightningDataModule, L.LightningModule]:
        """
        Initialize the experiment configuration from a given path.
        """
        loggers = Initializer.init_wandb(exp_config)
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb

        strategy = ModelParallelStrategy(  # <- uses FSDP2 under the hood
            data_parallel_size=exp_config.run.distributed.devices_per_node,  # shard across all 8 GPUs
            tensor_parallel_size=1,  # (= no TP)  just pure FSDP2
            # save_distributed_checkpoint=True,  # write one shard per rank
        )

        trainer = L.Trainer(
            max_epochs=exp_config.run.num_epochs,
            # Distributed training configuration
            limit_train_batches=exp_config.run.num_epochs
            * exp_config.run.steps_per_epoch,
            accelerator=exp_config.run.accelerator,
            devices=exp_config.run.distributed.devices_per_node,
            num_nodes=exp_config.run.distributed.num_nodes,
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
        datapack = exp_config.datapack.create_datapack(exp_config)
        lit_module = exp_config.module.create_module(exp_config.run)

        return trainer, datapack, lit_module
