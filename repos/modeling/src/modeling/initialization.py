from __future__ import annotations

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from wandb.sdk.wandb_run import Run

from modeling.config import ExperimentConfig, WandbConfig


class Initializer:
    """
    Initializer class to handle the setup of the modeling application.
    It initializes the experiment configuration, data pack, module, and logger.
    """

    @staticmethod
    def init_wandb(wandb_config: WandbConfig) -> WandbLogger:
        """
        Initialize a WandbLogger instance with the configuration.
        """
        return WandbLogger(project=wandb_config.project, name=wandb_config.name)

    @staticmethod
    def init_experiment(
        exp_config: ExperimentConfig,
    ) -> tuple[L.Trainer, L.LightningDataModule, L.LightningModule]:
        """
        Initialize the experiment configuration from a given path.
        """
        wandb_logger = Initializer.init_wandb(exp_config.metadata.wandb)
        wandb_experiment = wandb_logger.experiment
        assert isinstance(wandb_experiment, Run)
        wandb_experiment.config.update(exp_config.model_dump(serialize_as_any=True))
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb

        trainer = L.Trainer(
            max_epochs=exp_config.run.num_epochs,
            accelerator="auto",
            # Distributed training configuration
            devices=exp_config.run.distributed.devices_per_node,
            num_nodes=exp_config.run.distributed.num_nodes,
            strategy=exp_config.run.distributed.strategy,
            # Logging and checkpointing
            logger=wandb_logger,
            precision="bf16-true" if torch.cuda.is_bf16_supported() else "16-mixed",
        )
        datapack = exp_config.datapack.create_datapack(exp_config)
        lit_module = exp_config.module.create_module()

        return trainer, datapack, lit_module
